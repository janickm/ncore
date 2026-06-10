# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Argoverse 2 Sensor Dataset to NCore V4 converter."""

from __future__ import annotations

import json
import logging

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import click
import numpy as np
import tqdm

from pyquaternion import Quaternion
from upath import UPath

from ncore.impl.common.transformations import (
    HalfClosedInterval,
    MotionCompensator,
    se3_inverse,
    transform_bbox,
)
from ncore.impl.data.types import (
    BBox3,
    CuboidTrackObservation,
    LabelSource,
    OpenCVPinholeCameraModelParameters,
    ShutterType,
)
from ncore.impl.data.v4.components import (
    CameraSensorComponent,
    CuboidsComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    MasksComponent,
    PosesComponent,
    RadarSensorComponent,  # noqa: F401 -- imported for parity; AV2 has no radar
    SequenceComponentGroupsReader,
    SequenceComponentGroupsWriter,
)
from ncore.impl.data.v4.types import ComponentGroupAssignments
from ncore.impl.data_converter.base import FileBasedDataConverter, FileBasedDataConverterConfig
from tools.data_converter.argoverse2.utils import (
    AV2_CATEGORY_MAP,
    CAMERA_NAMES,
    LIDAR_NAMES,
    assign_lidar_units,
    list_log_ids,
    list_sensor_timestamps,
    read_annotations,
    read_city_se3_ego,
    read_ego_se3_sensor,
    read_intrinsics,
    read_lidar_sweep,
)
from tools.data_converter.cli import cli


# Argoverse 2 timestamps are nanoseconds; NCore V4 uses microseconds.
NS_PER_US = 1000


def _ns_to_us(value_ns: int) -> int:
    return int(value_ns) // NS_PER_US


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass(kw_only=True, slots=True)
class Argoverse2Converter4Config(FileBasedDataConverterConfig):
    """Configuration for Argoverse 2 to NCore V4 conversion."""

    split: str = "val"
    log_id: Optional[str] = None
    store_type: Literal["itar", "directory"] = "itar"
    component_group_profile: Literal["default", "separate-sensors", "separate-all"] = "separate-sensors"
    store_sequence_meta: bool = True


# -----------------------------------------------------------------------------
# Converter
# -----------------------------------------------------------------------------


class Argoverse2Converter4(FileBasedDataConverter):
    """Dataset preprocessing class for converting Argoverse 2 data to NCore V4.

    Sensor assumptions (sourced from the AV2 devkit and User Guide):

    - Cameras: 9 global-shutter cameras (7 ring + 2 stereo). Imagery is shipped
      already undistorted, so a pinhole model with zero distortion is exact. A
      single capture timestamp per image (no rolling-shutter metadata) ->
      ``ShutterType.GLOBAL``.
    - Lidar: two stacked Velodyne VLP-32C units (``up_lidar`` / ``down_lidar``,
      10 Hz). The source sweep is egomotion-compensated to the sweep reference
      timestamp and stored in the egovehicle frame. Real per-point timestamps are
      provided (``offset_ns``). We split points per unit by ``laser_number``,
      transform into each unit's sensor frame, and decompensate using the real
      per-point timestamps so NCore stores raw per-point-time directions. No
      structured lidar model is stored in this first version (``model_element``
      is ``None``); a derived VLP-32C model can be added as a follow-up.
    - Radar: AV2 has no radar.
    - Cuboid annotations: native to the egovehicle frame at the sweep reference
      time. Baked into the static ``world`` frame at conversion time using the
      exact ego pose, so their position does not depend on dynamic-pose
      interpolation (the ego moves ~1 m across a sweep, which otherwise shows up
      as a shift relative to the lidar).

    The first ego pose's ``city_SE3_egovehicle`` is stored as the static
    ``world -> world_global`` anchor, so ``world_global`` is the AV2 city frame.
    This keeps absolute coordinates recoverable for later HD-map alignment.
    """

    def __init__(self, config: Argoverse2Converter4Config) -> None:
        super().__init__(config)

        self.component_group_profile = config.component_group_profile
        self.store_type = config.store_type
        self.store_sequence_meta = config.store_sequence_meta

        self._split = config.split
        self._log_id = config.log_id

        self.logger = logging.getLogger(__name__)

    @property
    def _split_dir(self) -> UPath:
        return self.root_dir / self._split

    @staticmethod
    def get_sequence_ids(config: Argoverse2Converter4Config) -> List[str]:
        """Discover log IDs to convert."""
        if config.log_id is not None:
            return [config.log_id]
        split_dir = UPath(config.root_dir) / config.split
        return list_log_ids(split_dir)

    @staticmethod
    def from_config(config: Argoverse2Converter4Config) -> Argoverse2Converter4:
        return Argoverse2Converter4(config)

    def convert_sequence(self, sequence_id: str) -> None:
        """Convert a single Argoverse 2 log to NCore V4 format."""
        log_id = sequence_id
        log_dir = self._split_dir / log_id

        self.logger.info(f"Converting log {log_id} (split={self._split})")

        # --- Ego poses (egovehicle -> city) -----------------------------------
        pose_timestamps_ns, T_ego_city_all = read_city_se3_ego(log_dir)
        n_poses = len(pose_timestamps_ns)
        assert n_poses >= 2, f"Log has fewer than 2 ego poses: {n_poses}"

        # Full-precision ego pose lookup keyed by the raw nanosecond timestamp.
        # AV2 annotation timestamps map exactly onto these pose timestamps, so we
        # can look up the exact city_SE3_egovehicle pose for each annotated sweep
        # (used to bake cuboids into the static world_global / city frame).
        ego_city_by_ts_ns: Dict[int, np.ndarray] = {int(ts): T for ts, T in zip(pose_timestamps_ns, T_ego_city_all)}

        pose_timestamps_us = np.array([_ns_to_us(t) for t in pose_timestamps_ns], dtype=np.uint64)

        # AV2 ego poses are dense (some only nanoseconds apart), so the ns -> us
        # truncation can produce duplicate microsecond timestamps. The pose writer
        # requires strictly increasing timestamps, so keep the first pose for each
        # unique microsecond timestamp.
        pose_timestamps_us, unique_idx = np.unique(pose_timestamps_us, return_index=True)
        T_ego_city_all = T_ego_city_all[unique_idx]
        n_poses = len(pose_timestamps_us)
        assert n_poses >= 2, f"Log has fewer than 2 unique-microsecond ego poses: {n_poses}"

        # Anchor the first pose as world_global (the AV2 city frame); store all
        # poses relative to it so the relative poses are float32-safe.
        T_world_world_global = T_ego_city_all[0].copy()  # float64 for global accuracy
        T_world_global_inv = se3_inverse(T_world_world_global)
        T_rig_world_relative = (T_world_global_inv @ T_ego_city_all).astype(np.float32)

        # --- Static sensor extrinsics (sensor -> ego) -------------------------
        ego_se3_sensor = read_ego_se3_sensor(log_dir)

        # --- Determine active sensors -----------------------------------------
        camera_ids = self.get_active_camera_ids(list(CAMERA_NAMES))
        lidar_ids = self.get_active_lidar_ids(list(LIDAR_NAMES))
        radar_ids = self.get_active_radar_ids([])  # AV2 has no radar

        # --- Sequence time interval -------------------------------------------
        all_ts_us: List[int] = [int(pose_timestamps_us[0]), int(pose_timestamps_us[-1])]
        lidar_ts_ns = list_sensor_timestamps(log_dir, "lidar")
        if lidar_ts_ns:
            all_ts_us += [_ns_to_us(lidar_ts_ns[0]), _ns_to_us(lidar_ts_ns[-1])]
        for cam_id in camera_ids:
            cam_ts_ns = list_sensor_timestamps(log_dir, "cameras", cam_id)
            if cam_ts_ns:
                all_ts_us += [_ns_to_us(cam_ts_ns[0]), _ns_to_us(cam_ts_ns[-1])]

        seq_start_us = min(all_ts_us)
        seq_end_us = max(all_ts_us)

        # Extend pose timeline to cover the sequence interval, extrapolating with a
        # constant-velocity assumption so lidar decompensation near the boundaries
        # has real motion to invert (mirrors the nuScenes converter).
        T_rig_world_relative, pose_timestamps_us = self._extend_pose_timeline(
            T_rig_world_relative, pose_timestamps_us, seq_start_us, seq_end_us
        )

        sequence_timestamp_interval_us = HalfClosedInterval.from_start_end(seq_start_us, seq_end_us)

        # --- Component group assignments --------------------------------------
        component_groups = ComponentGroupAssignments.create(
            camera_ids=camera_ids,
            lidar_ids=lidar_ids,
            radar_ids=radar_ids,
            point_clouds_ids=[],
            camera_labels_ids=[],
            profile=self.component_group_profile,
        )

        # --- Create writer ----------------------------------------------------
        store_writer = SequenceComponentGroupsWriter(
            output_dir_path=self.output_dir / log_id,
            store_base_name=log_id,
            sequence_id=log_id,
            sequence_timestamp_interval_us=sequence_timestamp_interval_us,
            store_type=self.store_type,
            generic_meta_data={
                "source_dataset": "argoverse2",
                "argoverse2_split": self._split,
                "argoverse2_log_id": log_id,
            },
        )

        poses_writer = store_writer.register_component_writer(
            PosesComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.poses_component_group,
            generic_meta_data={
                "calibration_type": "argoverse2:egovehicle_SE3_sensor",
                "egomotion_type": "argoverse2:city_SE3_egovehicle",
            },
        )
        intrinsics_writer = store_writer.register_component_writer(
            IntrinsicsComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.intrinsics_component_group,
        )
        masks_writer = store_writer.register_component_writer(
            MasksComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.masks_component_group,
        )

        # --- Store ego poses --------------------------------------------------
        poses_writer.store_dynamic_pose(
            source_frame_id="rig",
            target_frame_id="world",
            poses=T_rig_world_relative,
            timestamps_us=pose_timestamps_us,
        )
        poses_writer.store_static_pose(
            source_frame_id="world",
            target_frame_id="world_global",
            pose=T_world_world_global,
        )

        # --- Decode sensors ---------------------------------------------------
        if lidar_ids:
            self._decode_lidars(
                log_dir=log_dir,
                store_writer=store_writer,
                poses_writer=poses_writer,
                component_groups=component_groups,
                lidar_ids=lidar_ids,
                ego_se3_sensor=ego_se3_sensor,
                T_rig_world_relative=T_rig_world_relative,
                pose_timestamps_us=pose_timestamps_us,
            )

        self._decode_cameras(
            log_dir=log_dir,
            store_writer=store_writer,
            poses_writer=poses_writer,
            intrinsics_writer=intrinsics_writer,
            masks_writer=masks_writer,
            component_groups=component_groups,
            camera_ids=camera_ids,
            ego_se3_sensor=ego_se3_sensor,
        )

        self._decode_cuboids(
            log_dir=log_dir,
            store_writer=store_writer,
            component_groups=component_groups,
            ego_city_by_ts_ns=ego_city_by_ts_ns,
            T_world_global_inv=T_world_global_inv,
        )

        # --- Finalize ---------------------------------------------------------
        ncore_4_paths = store_writer.finalize()

        if self.store_sequence_meta:
            sequence_component_reader = SequenceComponentGroupsReader(ncore_4_paths)
            sequence_meta_path = self.output_dir / log_id / f"{sequence_component_reader.sequence_id}.json"
            with sequence_meta_path.open("w") as f:
                json.dump(sequence_component_reader.get_sequence_meta().to_dict(), f, indent=2)
            self.logger.info(f"Wrote sequence meta data {str(sequence_meta_path)}")

    # -------------------------------------------------------------------------
    # Pose timeline
    # -------------------------------------------------------------------------

    @staticmethod
    def _extend_pose_timeline(
        T_rig_world_relative: np.ndarray,
        pose_timestamps_us: np.ndarray,
        seq_start_us: int,
        seq_end_us: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extend the relative pose timeline to cover [seq_start, seq_end].

        Boundaries are extrapolated with a constant-velocity assumption so the
        first/last lidar sweeps have real ego motion to decompensate against.
        """
        if seq_start_us < int(pose_timestamps_us[0]):
            T_0 = T_rig_world_relative[0]
            T_1 = T_rig_world_relative[1]
            T_delta_inv = se3_inverse(T_1) @ T_0
            T_boundary = (T_0 @ T_delta_inv).astype(np.float32)
            T_rig_world_relative = np.concatenate([T_boundary[np.newaxis], T_rig_world_relative], axis=0)
            pose_timestamps_us = np.concatenate([np.array([seq_start_us], dtype=np.uint64), pose_timestamps_us])

        if seq_end_us > int(pose_timestamps_us[-1]):
            T_n1 = T_rig_world_relative[-2]
            T_n = T_rig_world_relative[-1]
            T_delta = se3_inverse(T_n1) @ T_n
            T_boundary = (T_n @ T_delta).astype(np.float32)
            T_rig_world_relative = np.concatenate([T_rig_world_relative, T_boundary[np.newaxis]], axis=0)
            pose_timestamps_us = np.concatenate([pose_timestamps_us, np.array([seq_end_us], dtype=np.uint64)])

        return T_rig_world_relative, pose_timestamps_us

    # -------------------------------------------------------------------------
    # Lidar
    # -------------------------------------------------------------------------

    def _decode_lidars(
        self,
        log_dir: UPath,
        store_writer: SequenceComponentGroupsWriter,
        poses_writer: PosesComponent.Writer,
        component_groups: ComponentGroupAssignments,
        lidar_ids: List[str],
        ego_se3_sensor: Dict[str, np.ndarray],
        T_rig_world_relative: np.ndarray,
        pose_timestamps_us: np.ndarray,
    ) -> None:
        """Decode and store the two stacked VLP-32C lidars individually.

        AV2 lidar points are egomotion-compensated to the sweep reference time and
        provided in the egovehicle frame. For each unit we map points into the
        unit's own sensor frame (via the static extrinsic) and decompensate using
        the real per-point timestamps to recover raw per-point-time directions.

        Because the sensor extrinsic is static, the decompensation commutes with
        the ego->sensor transform, so the result is independent of whether AV2
        applied ego-motion before or after the sensor transform.
        """
        sweep_ts_ns = list_sensor_timestamps(log_dir, "lidar")
        if not sweep_ts_ns:
            self.logger.warning("No lidar sweeps found")
            return

        # Static extrinsics + per-unit writers + per-unit motion compensators.
        T_unit_ego: Dict[str, np.ndarray] = {}
        lidar_writers: Dict[str, LidarSensorComponent.Writer] = {}
        compensators: Dict[str, MotionCompensator] = {}

        for unit in lidar_ids:
            T_unit_ego[unit] = ego_se3_sensor[unit].astype(np.float32)
            poses_writer.store_static_pose(source_frame_id=unit, target_frame_id="rig", pose=T_unit_ego[unit])
            lidar_writers[unit] = store_writer.register_component_writer(
                LidarSensorComponent.Writer,
                component_instance_name=unit,
                group_name=component_groups.lidar_component_groups.get(unit),
                generic_meta_data={"sensor_model": "Velodyne VLP-32C"},
            )
            compensators[unit] = MotionCompensator.from_sensor_rig(
                sensor_id=unit,
                T_sensor_rig=T_unit_ego[unit],
                T_rig_worlds=T_rig_world_relative,
                T_rig_worlds_timestamps_us=pose_timestamps_us,
            )

        # Determine the laser_number -> unit labelling once from the first sweep.
        # The split is a fixed physical property of the two stacked sensors, so we
        # resolve it once (robustly, from extrinsic geometry) and reuse it for every
        # sweep rather than re-deciding per frame.
        T_up = ego_se3_sensor["up_lidar"] if "up_lidar" in ego_se3_sensor else np.eye(4)
        T_down = ego_se3_sensor["down_lidar"] if "down_lidar" in ego_se3_sensor else np.eye(4)
        first_sweep = read_lidar_sweep(log_dir / "sensors" / "lidar" / f"{sweep_ts_ns[0]}.feather")
        first_masks = assign_lidar_units(first_sweep.laser_number, first_sweep.xyz, T_up, T_down)
        # Map the decision to a laser_number threshold test (lo half == laser_number < 32).
        lo_is_up = bool(first_masks["up_lidar"][first_sweep.laser_number < 4].all())
        unit_for_lo = "up_lidar" if lo_is_up else "down_lidar"
        unit_for_hi = "down_lidar" if lo_is_up else "up_lidar"
        self.logger.info(f"Lidar unit split: laser_number<32 -> {unit_for_lo}, >=32 -> {unit_for_hi}")

        for i, ts_ns in enumerate(tqdm.tqdm(sweep_ts_ns, desc="Process lidar")):
            sweep = read_lidar_sweep(log_dir / "sensors" / "lidar" / f"{ts_ns}.feather")

            # Per-point absolute time = sweep_ts + offset; convert to microseconds.
            point_ts_us = ((sweep.timestamp_ns + sweep.offset_ns) // NS_PER_US).astype(np.uint64)

            frame_start_us = int(point_ts_us.min())
            frame_end_us = int(point_ts_us.max())
            if frame_end_us <= frame_start_us:
                frame_end_us = frame_start_us + 1

            lo = sweep.laser_number < 32
            unit_masks = {unit_for_lo: lo, unit_for_hi: ~lo}

            for unit in lidar_ids:
                mask = unit_masks[unit]
                if not mask.any():
                    continue

                xyz_ego = sweep.xyz[mask]
                ts_unit = point_ts_us[mask]
                intensity_unit = sweep.intensity[mask]

                # ego -> unit sensor frame
                T_ego_unit = se3_inverse(T_unit_ego[unit])
                xyz_sensor = (T_ego_unit[:3, :3] @ xyz_ego.T).T + T_ego_unit[:3, 3]

                # Decompensate to per-point-time sensor frame (raw measurements).
                xyz_raw = compensators[unit].motion_decompensate_points(
                    sensor_id=unit,
                    xyz_sensorend=xyz_sensor.astype(np.float32),
                    timestamp_us=ts_unit,
                    frame_start_timestamp_us=frame_start_us,
                    frame_end_timestamp_us=frame_end_us,
                )

                distance_m = np.linalg.norm(xyz_raw, axis=1).astype(np.float32)
                direction = np.zeros_like(xyz_raw)
                nonzero = distance_m > 0
                direction[nonzero] = xyz_raw[nonzero] / distance_m[nonzero, np.newaxis]

                lidar_writers[unit].store_frame(
                    direction=direction.astype(np.float32),
                    timestamp_us=ts_unit,
                    model_element=None,
                    distance_m=distance_m.reshape(1, -1),
                    intensity=intensity_unit.reshape(1, -1),
                    frame_timestamps_us=np.array([frame_start_us, frame_end_us], dtype=np.uint64),
                    generic_data={},
                    generic_meta_data={},
                )

    # -------------------------------------------------------------------------
    # Cameras
    # -------------------------------------------------------------------------

    def _decode_cameras(
        self,
        log_dir: UPath,
        store_writer: SequenceComponentGroupsWriter,
        poses_writer: PosesComponent.Writer,
        intrinsics_writer: IntrinsicsComponent.Writer,
        masks_writer: MasksComponent.Writer,
        component_groups: ComponentGroupAssignments,
        camera_ids: List[str],
        ego_se3_sensor: Dict[str, np.ndarray],
    ) -> None:
        """Decode and store all camera frames.

        AV2 cameras are global shutter and imagery is shipped undistorted, so the
        stored model is a pinhole with zero distortion coefficients.
        """
        intrinsics = read_intrinsics(log_dir)

        for cam_id in camera_ids:
            cam_ts_ns = list_sensor_timestamps(log_dir, "cameras", cam_id)
            if not cam_ts_ns:
                self.logger.warning(f"No data for camera {cam_id}")
                continue

            self.logger.info(f"Processing camera {cam_id}")

            # Extrinsic: camera -> rig (ego)
            T_cam_rig = ego_se3_sensor[cam_id].astype(np.float32)
            poses_writer.store_static_pose(source_frame_id=cam_id, target_frame_id="rig", pose=T_cam_rig)

            intr = intrinsics[cam_id]
            intrinsics_writer.store_camera_intrinsics(
                camera_id=cam_id,
                camera_model_parameters=OpenCVPinholeCameraModelParameters(
                    resolution=np.array([intr.width, intr.height], dtype=np.uint64),
                    shutter_type=ShutterType.GLOBAL,
                    external_distortion_parameters=None,
                    principal_point=np.array([intr.cx, intr.cy], dtype=np.float32),
                    focal_length=np.array([intr.fx, intr.fy], dtype=np.float32),
                    radial_coeffs=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    tangential_coeffs=np.array([0.0, 0.0], dtype=np.float32),
                    thin_prism_coeffs=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                ),
            )

            masks_writer.store_camera_masks(camera_id=cam_id, mask_images={})

            camera_writer = store_writer.register_component_writer(
                CameraSensorComponent.Writer,
                component_instance_name=cam_id,
                group_name=component_groups.camera_component_groups.get(cam_id),
                generic_meta_data={},
            )

            cam_dir = log_dir / "sensors" / "cameras" / cam_id
            for ts_ns in tqdm.tqdm(cam_ts_ns, desc=f"Process {cam_id}"):
                image_path = cam_dir / f"{ts_ns}.jpg"
                with image_path.open("rb") as f:
                    image_binary = f.read()

                frame_ts = _ns_to_us(ts_ns)
                camera_writer.store_frame(
                    image_binary_data=image_binary,
                    image_format="jpeg",
                    frame_timestamps_us=np.array([frame_ts, frame_ts], dtype=np.uint64),
                    generic_data={},
                    generic_meta_data={},
                )

        self.logger.info(f"Processed {len(camera_ids)} cameras")

    # -------------------------------------------------------------------------
    # Cuboid annotations
    # -------------------------------------------------------------------------

    def _decode_cuboids(
        self,
        log_dir: UPath,
        store_writer: SequenceComponentGroupsWriter,
        component_groups: ComponentGroupAssignments,
        ego_city_by_ts_ns: Dict[int, np.ndarray],
        T_world_global_inv: np.ndarray,
    ) -> None:
        """Decode AV2 3D annotations and store as cuboid track observations.

        AV2 cuboids are native to the egovehicle frame *at the sweep reference
        timestamp*. The egovehicle moves up to ~1 m across a single ~100 ms sweep,
        so referencing cuboids to the dynamic ``rig`` frame makes their rendered
        position depend on how a downstream consumer interpolates the rig pose for
        a given timestamp -- which produces a visible shift relative to the lidar.

        To avoid that, we bake each cuboid into the static ``world`` frame at
        conversion time using the exact ``city_SE3_egovehicle`` pose for the
        annotated sweep. ``world`` is the local frame anchored at the first ego
        pose (``world_global`` is the AV2 city frame), so storing here is both
        time-independent and float32-safe. This mirrors the nuScenes converter,
        which stores cuboids in a static global frame.
        """
        annotations_path = log_dir / "annotations.feather"
        if not annotations_path.exists():
            self.logger.info("No annotations.feather found (test split)")
            return

        cols = read_annotations(log_dir)
        n = len(cols["category"])

        cuboid_observations: List[CuboidTrackObservation] = []
        missing_pose = 0
        for i in tqdm.tqdm(range(n), total=n, desc="Process cuboids"):
            category = str(cols["category"][i])
            if category not in AV2_CATEGORY_MAP:
                continue

            ts_ns = int(cols["timestamp_ns"][i])
            timestamp_us = _ns_to_us(ts_ns)

            T_ego_city = ego_city_by_ts_ns.get(ts_ns)
            if T_ego_city is None:
                missing_pose += 1
                continue

            yaw = Quaternion(cols["qw"][i], cols["qx"][i], cols["qy"][i], cols["qz"][i]).yaw_pitch_roll[0]

            # Cuboid in the egovehicle frame at the sweep reference time.
            # AV2: length_m -> x extent, width_m -> y extent, height_m -> z extent.
            bbox_ego = np.array(
                [
                    cols["tx_m"][i],
                    cols["ty_m"][i],
                    cols["tz_m"][i],
                    cols["length_m"][i],
                    cols["width_m"][i],
                    cols["height_m"][i],
                    0.0,  # roll
                    0.0,  # pitch
                    yaw,
                ],
                dtype=np.float64,
            )

            # ego -> city -> world (static local anchor).
            T_world_ego = T_world_global_inv @ T_ego_city
            bbox_world = transform_bbox(bbox_ego, T_world_ego)
            bbox3 = BBox3.from_array(bbox_world.astype(np.float32))

            cuboid_observations.append(
                CuboidTrackObservation(
                    track_id=str(cols["track_uuid"][i]),
                    class_id=AV2_CATEGORY_MAP[category],
                    timestamp_us=timestamp_us,
                    reference_frame_id="world",
                    reference_frame_timestamp_us=timestamp_us,
                    bbox3=bbox3,
                    source=LabelSource.EXTERNAL,
                )
            )

        if missing_pose:
            self.logger.warning(f"Skipped {missing_pose} cuboids with no matching ego pose timestamp")

        if cuboid_observations:
            store_writer.register_component_writer(
                CuboidsComponent.Writer,
                "default",
                component_groups.cuboid_track_observations_component_group,
            ).store_observations(cuboid_observations)
            self.logger.info(f"Stored {len(cuboid_observations)} cuboid observations")
        else:
            self.logger.info("No mapped cuboid annotations found")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


@cli.command(name="argoverse2-v4")
@click.option(
    "--split",
    type=str,
    default="val",
    show_default=True,
    help="Argoverse 2 split directory under --root-dir (e.g. train, val, test)",
)
@click.option(
    "--log-id",
    type=str,
    default=None,
    help="Convert only the log with this ID (defaults to all logs in the split)",
)
@click.option(
    "--store-type",
    type=click.Choice(["itar", "directory"], case_sensitive=False),
    default="itar",
    show_default=True,
    help="Output store type",
)
@click.option(
    "component_group_profile",
    "--profile",
    type=click.Choice(["default", "separate-sensors", "separate-all"], case_sensitive=False),
    default="separate-sensors",
    show_default=True,
    help="Output profile for component group assignment",
)
@click.option(
    "store_sequence_meta",
    "--sequence-meta/--no-sequence-meta",
    default=True,
    help="Generate sequence meta-data JSON?",
)
@click.pass_context
def argoverse2_v4(ctx, split, log_id, **kwargs):
    """Argoverse 2 Sensor Dataset conversion (V4 format)"""

    config = Argoverse2Converter4Config(**{**vars(ctx.obj), "split": split, "log_id": log_id, **kwargs})

    Argoverse2Converter4.convert(config)

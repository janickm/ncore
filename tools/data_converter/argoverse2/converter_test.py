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

"""Integration tests for the Argoverse 2 data converter (V4 format).

Requires the AV2_DIR environment variable pointing to an Argoverse 2 Sensor
Dataset root directory organised as ``{AV2_DIR}/{split}/{log_id}/...``.

Set AV2_SPLIT to override the default split (``val``). The first log in the split
is used for testing.
"""

import os
import tempfile
import unittest

from typing import Literal, cast

import numpy as np

from parameterized import parameterized_class
from upath import UPath

from ncore.impl.data.types import OpenCVPinholeCameraModelParameters, ShutterType
from ncore.impl.data.v4.components import (
    CameraSensorComponent,
    CuboidsComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    PosesComponent,
    RadarSensorComponent,
    SequenceComponentGroupsReader,
)
from tools.data_converter.argoverse2.converter import Argoverse2Converter4, Argoverse2Converter4Config
from tools.data_converter.argoverse2.utils import CAMERA_NAMES, LIDAR_NAMES, list_log_ids


@parameterized_class(
    ("store_type",),
    [
        ("itar",),
        ("directory",),
    ],
)
class TestArgoverse2Converter(unittest.TestCase):
    """Integration tests for the Argoverse 2 data converter.

    Requires AV2_DIR environment variable pointing to an Argoverse 2 Sensor Dataset
    root. Uses the first log in the split for testing.
    """

    store_type: Literal["itar", "directory"]

    @classmethod
    def setUpClass(cls):
        cls.av2_dir = os.environ.get("AV2_DIR")
        if cls.av2_dir is None:
            raise unittest.SkipTest("AV2_DIR not set -- skipping Argoverse 2 integration tests")

        cls.split = os.environ.get("AV2_SPLIT", "val")

        log_ids = list_log_ids(UPath(cls.av2_dir) / cls.split)
        assert log_ids, f"No logs found under {cls.av2_dir}/{cls.split}"
        cls.log_id = log_ids[0]

        cls._tempdir = tempfile.TemporaryDirectory(prefix="argoverse2_test_")
        cls.output_dir = cls._tempdir.name

        config = Argoverse2Converter4Config(
            root_dir=cls.av2_dir,
            output_dir=cls.output_dir,
            no_cameras=False,
            camera_ids=None,
            no_lidars=False,
            lidar_ids=None,
            no_radars=False,
            radar_ids=None,
            verbose=False,
            debug=False,
            debug_port=5678,
            split=cls.split,
            log_id=cls.log_id,
            store_type=cls.store_type,
            component_group_profile="separate-sensors",
            store_sequence_meta=True,
        )
        Argoverse2Converter4.convert(config)

        seq_dirs = [d for d in UPath(cls.output_dir).iterdir() if d.is_dir()]
        assert len(seq_dirs) == 1, f"Expected 1 sequence dir, found {len(seq_dirs)}: {seq_dirs}"
        cls.seq_dir = seq_dirs[0]

        meta_files = list(cls.seq_dir.glob("*.json"))
        assert len(meta_files) == 1, f"Expected 1 meta JSON, found {len(meta_files)}"
        cls.reader = SequenceComponentGroupsReader([meta_files[0]])

    @classmethod
    def tearDownClass(cls):
        cls._tempdir.cleanup()

    # --- Poses ----------------------------------------------------------------

    def test_sequence_has_dynamic_rig_to_world_pose(self):
        poses_readers = self.reader.open_component_readers(PosesComponent.Reader)
        self.assertEqual(len(poses_readers), 1)
        poses_reader = list(poses_readers.values())[0]

        poses, timestamps = poses_reader.get_dynamic_pose("rig", "world")
        self.assertEqual(poses.shape[1:], (4, 4))
        self.assertGreater(poses.shape[0], 0)
        self.assertEqual(timestamps.shape[0], poses.shape[0])

    def test_sequence_has_static_world_to_world_global(self):
        """world_global is the AV2 city frame; verify the static anchor exists."""
        poses_readers = self.reader.open_component_readers(PosesComponent.Reader)
        poses_reader = list(poses_readers.values())[0]

        static_poses = dict(poses_reader.get_static_poses())
        self.assertIn(("world", "world_global"), static_poses)
        self.assertEqual(static_poses[("world", "world_global")].shape, (4, 4))

    def test_first_real_pose_near_identity(self):
        """The anchored ego pose is stored as relative identity in the trajectory.

        The first pose's city_SE3_egovehicle is the world_global anchor, so its
        relative rig -> world pose must be (near) identity. Boundary extrapolation
        may prepend an extra pose, so we locate the identity pose rather than
        assuming a fixed index.
        """
        poses_readers = self.reader.open_component_readers(PosesComponent.Reader)
        poses_reader = list(poses_readers.values())[0]

        poses, _ = poses_reader.get_dynamic_pose("rig", "world")
        deviations = np.linalg.norm(poses - np.eye(4, dtype=np.float32), axis=(1, 2))
        np.testing.assert_array_almost_equal(poses[int(np.argmin(deviations))], np.eye(4, dtype=np.float32), decimal=3)

    # --- Cameras --------------------------------------------------------------

    def test_nine_cameras_exist(self):
        camera_readers = self.reader.open_component_readers(CameraSensorComponent.Reader)
        self.assertEqual(set(camera_readers.keys()), set(CAMERA_NAMES))
        for cam_id, cam_reader in camera_readers.items():
            self.assertGreater(cam_reader.frames_count, 0, f"{cam_id} should have frames")

    def test_camera_intrinsics_global_shutter_zero_distortion(self):
        intrinsics_readers = self.reader.open_component_readers(IntrinsicsComponent.Reader)
        self.assertEqual(len(intrinsics_readers), 1)
        intrinsics_reader = list(intrinsics_readers.values())[0]

        for cam_id in CAMERA_NAMES:
            params = intrinsics_reader.get_camera_model_parameters(cam_id)
            self.assertIsInstance(params, OpenCVPinholeCameraModelParameters)
            params = cast(OpenCVPinholeCameraModelParameters, params)
            self.assertEqual(params.shutter_type, ShutterType.GLOBAL)
            np.testing.assert_array_equal(params.radial_coeffs, np.zeros(6, dtype=np.float32))
            np.testing.assert_array_equal(params.tangential_coeffs, np.zeros(2, dtype=np.float32))
            self.assertTrue(np.all(params.focal_length > 0))

    def test_camera_extrinsics_stored_as_static_poses(self):
        poses_readers = self.reader.open_component_readers(PosesComponent.Reader)
        poses_reader = list(poses_readers.values())[0]

        static_poses = dict(poses_reader.get_static_poses())
        for cam_id in CAMERA_NAMES:
            self.assertIn((cam_id, "rig"), static_poses)

    # --- Lidar ----------------------------------------------------------------

    def test_two_lidar_units_exist(self):
        lidar_readers = self.reader.open_component_readers(LidarSensorComponent.Reader)
        self.assertEqual(set(lidar_readers.keys()), set(LIDAR_NAMES))
        for lidar_id, lidar_reader in lidar_readers.items():
            self.assertGreater(lidar_reader.frames_count, 0, f"{lidar_id} should have frames")

    def test_lidar_extrinsics_stored_as_static_poses(self):
        poses_readers = self.reader.open_component_readers(PosesComponent.Reader)
        poses_reader = list(poses_readers.values())[0]

        static_poses = dict(poses_reader.get_static_poses())
        for lidar_id in LIDAR_NAMES:
            self.assertIn((lidar_id, "rig"), static_poses)

    def test_lidar_directions_unit_norm(self):
        lidar_readers = self.reader.open_component_readers(LidarSensorComponent.Reader)
        lidar_reader = lidar_readers["up_lidar"]
        ts = int(lidar_reader.frames_timestamps_us[0, 1])  # end-of-frame timestamp key
        direction = lidar_reader.get_frame_ray_bundle_data(ts, "direction")
        norms = np.linalg.norm(direction, axis=1)
        # Zero-distance rays may have zero direction; check the populated ones.
        nonzero = norms > 0
        np.testing.assert_allclose(norms[nonzero], 1.0, atol=1e-4)

    def test_lidar_unit_split_recovered_from_geometry(self):
        """The two units carry comparable point counts (~half the sweep each).

        Each VLP-32C contributes 32 of the 64 beams, so a correct split yields
        roughly balanced point counts per unit (allowing for differing FOV
        occupancy).
        """
        lidar_readers = self.reader.open_component_readers(LidarSensorComponent.Reader)
        counts = {}
        for unit in ("up_lidar", "down_lidar"):
            reader = lidar_readers[unit]
            ts = int(reader.frames_timestamps_us[0, 1])
            counts[unit] = len(reader.get_frame_ray_bundle_data(ts, "direction"))
        ratio = min(counts.values()) / max(counts.values())
        self.assertGreater(ratio, 0.5, f"Lidar unit point counts unbalanced: {counts}")

    # --- No radar -------------------------------------------------------------

    def test_no_radar(self):
        radar_readers = self.reader.open_component_readers(RadarSensorComponent.Reader)
        self.assertEqual(len(radar_readers), 0)

    # --- Cuboids --------------------------------------------------------------

    def test_cuboids_in_static_world_frame(self):
        """Cuboids are baked into the static ``world`` frame (time-independent)."""
        cuboid_readers = self.reader.open_component_readers(CuboidsComponent.Reader)
        if not cuboid_readers:
            self.skipTest("No cuboids (test split)")
        cuboid_reader = list(cuboid_readers.values())[0]
        observations = list(cuboid_reader.get_observations())
        self.assertGreater(len(observations), 0)
        for obs in observations[:50]:
            self.assertEqual(obs.reference_frame_id, "world")

    def test_cuboids_align_with_lidar(self):
        """A reasonable fraction of lidar points fall inside annotated cuboids.

        This is the regression guard for the timestamp/frame shift: if cuboids were
        mis-referenced relative to the lidar, almost no points would land inside
        them. We transform first-frame lidar points (sensor -> world) and count
        points inside the cuboids active at that sweep.
        """
        from ncore.impl.common.transformations import is_within_3d_bboxes

        cuboid_readers = self.reader.open_component_readers(CuboidsComponent.Reader)
        if not cuboid_readers:
            self.skipTest("No cuboids (test split)")

        poses_reader = list(self.reader.open_component_readers(PosesComponent.Reader).values())[0]
        lidar_reader = self.reader.open_component_readers(LidarSensorComponent.Reader)["up_lidar"]
        frame_start_us, frame_end_us = (int(v) for v in lidar_reader.frames_timestamps_us[0])
        ts = frame_end_us  # reader frame key is the end-of-frame timestamp

        # The cuboid reference timestamp is the AV2 sweep reference time, which is
        # the start of the point window (offset_ns runs forward from it).
        cuboid_ts = frame_start_us

        # Each lidar point is in its own per-point-time sensor frame; transform via
        # sensor -> rig (static) and rig -> world at the point's own timestamp.
        static = dict(poses_reader.get_static_poses())
        T_up_rig = static[("up_lidar", "rig")]
        rig_poses, pose_ts = poses_reader.get_dynamic_pose("rig", "world")

        direction = lidar_reader.get_frame_ray_bundle_data(ts, "direction")
        distance = np.asarray(lidar_reader._get_ray_bundle_returns_group(ts)["distance_m"])[0]
        point_ts = lidar_reader.get_frame_ray_bundle_data(ts, "timestamp_us").astype(np.int64)
        valid = np.isfinite(distance) & (distance > 0)
        pts_sensor = direction[valid] * distance[valid, None]
        pts_rig = (T_up_rig[:3, :3] @ pts_sensor.T).T + T_up_rig[:3, 3]
        # Per-point rig -> world using the nearest stored pose (poses are dense).
        nearest = np.searchsorted(pose_ts.astype(np.int64), point_ts[valid]).clip(0, len(rig_poses) - 1)
        T_pts = rig_poses[nearest]
        pts_world = np.einsum("nij,nj->ni", T_pts[:, :3, :3], pts_rig) + T_pts[:, :3, 3]

        observations = list(list(cuboid_readers.values())[0].get_observations())
        # cuboids active at this sweep (cuboid ref ts == sweep start)
        active = [o for o in observations if abs(o.reference_frame_timestamp_us - cuboid_ts) < 2000]
        self.assertGreater(len(active), 0, "no cuboids active at first lidar frame")
        boxes = np.stack([o.bbox3.to_array() for o in active])
        inside = is_within_3d_bboxes(pts_world.astype(np.float64), boxes.astype(np.float64))
        n_inside = int(inside.any(axis=1).sum())
        self.assertGreater(
            n_inside, 50, f"only {n_inside} lidar points inside any cuboid -- likely a frame/timestamp shift"
        )

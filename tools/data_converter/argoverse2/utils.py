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

"""Argoverse 2 specific utilities for the NCore V4 converter.

This module reads the Argoverse 2 Sensor Dataset directly from its on-disk
Apache Feather files using ``pyarrow`` only, deliberately avoiding the heavy
``av2`` devkit (which pulls in torch, kornia, numba, polars and PyAV). The only
extra dependency is ``pyquaternion`` for the scalar-first (wxyz) quaternion to
rotation-matrix conversion, which is already used elsewhere in this package.

Reference (sourced from github.com/argoverse/av2-api and the AV2 User Guide):

- Lidar sweeps are *egomotion-compensated* to the sweep reference timestamp and
  stored in the **egovehicle** frame (not the individual sensor frame). The
  feather columns are ``x, y, z, intensity, laser_number, offset_ns``.
  Per-point absolute time is ``sweep_timestamp_ns + offset_ns``.
- Cameras are **global shutter** and the released imagery is **already
  undistorted**, so a pinhole model with zero distortion is exact.
- All quaternions are scalar-first ``(qw, qx, qy, qz)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pyarrow.feather as feather

from pyquaternion import Quaternion
from upath import UPath


# --- Feather reading (no pandas) -----------------------------------------------
# We read Arrow tables directly and pull columns out as numpy arrays. This avoids
# pulling pandas into the dependency closure (pyarrow.read_feather would default to
# a pandas DataFrame).


def _read_columns(path: UPath) -> Dict[str, np.ndarray]:
    """Read a feather file into a ``column_name -> numpy array`` mapping."""
    table = feather.read_table(str(path))
    return {name: table.column(name).to_numpy(zero_copy_only=False) for name in table.column_names}


# --- Sensor ID mappings --------------------------------------------------------
# Argoverse 2 sensor names are already descriptive; we keep them verbatim as the
# NCore sensor IDs so that any downstream alignment with AV2 map / metadata stays
# unambiguous.

# All nine global-shutter cameras (7 ring + 2 stereo).
CAMERA_NAMES: List[str] = [
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_side_left",
    "ring_side_right",
    "ring_rear_left",
    "ring_rear_right",
    "stereo_front_left",
    "stereo_front_right",
]

# The two stacked Velodyne VLP-32C units.
LIDAR_NAMES: List[str] = ["up_lidar", "down_lidar"]

# Number of beams per VLP-32C unit. laser_number spans [0, 63] across both units.
VLP32C_N_BEAMS: int = 32

# AV2 ships no radar.

# --- Annotation taxonomy -------------------------------------------------------
# Argoverse 2 3D cuboid categories (the 30-class `AnnotationCategories` taxonomy)
# mapped to NCore class IDs. AV2 category strings are upper snake-case.
AV2_CATEGORY_MAP: Dict[str, str] = {
    "REGULAR_VEHICLE": "car",
    "LARGE_VEHICLE": "truck",
    "BOX_TRUCK": "truck",
    "TRUCK": "truck",
    "TRUCK_CAB": "truck",
    "VEHICULAR_TRAILER": "trailer",
    "SCHOOL_BUS": "bus",
    "ARTICULATED_BUS": "bus",
    "BUS": "bus",
    "MESSAGE_BOARD_TRAILER": "trailer",
    "RAILED_VEHICLE": "vehicle",
    "MOTORCYCLE": "motorcycle",
    "MOTORCYCLIST": "motorcyclist",
    "BICYCLE": "bicycle",
    "BICYCLIST": "bicyclist",
    "WHEELED_DEVICE": "wheeled_device",
    "WHEELED_RIDER": "wheeled_rider",
    "PEDESTRIAN": "pedestrian",
    "OFFICIAL_SIGNALER": "pedestrian",
    "STROLLER": "stroller",
    "WHEELCHAIR": "wheelchair",
    "DOG": "animal",
    "ANIMAL": "animal",
    "CONSTRUCTION_CONE": "traffic_cone",
    "CONSTRUCTION_BARREL": "barrier",
    "STOP_SIGN": "stop_sign",
    "BOLLARD": "bollard",
    "SIGN": "sign",
    "MOBILE_PEDESTRIAN_CROSSING_SIGN": "sign",
    "TRAFFIC_LIGHT_TRAILER": "trailer",
}


# --- Pose / quaternion helpers -------------------------------------------------


def se3_from_qwxyz_t(qw: float, qx: float, qy: float, qz: float, tx: float, ty: float, tz: float) -> np.ndarray:
    """Build a 4x4 SE(3) matrix from a scalar-first quaternion and translation.

    Argoverse 2 stores all rotations as ``(qw, qx, qy, qz)``.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Quaternion(qw, qx, qy, qz).rotation_matrix
    T[:3, 3] = (tx, ty, tz)
    return T


# --- Dataset layout / feather readers ------------------------------------------


@dataclass(frozen=True)
class CameraIntrinsics:
    """Parsed pinhole intrinsics for a single AV2 camera."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def list_log_ids(split_dir: UPath) -> List[str]:
    """Return the sorted log IDs (sub-directory names) under a split directory."""
    return sorted(p.name for p in split_dir.iterdir() if p.is_dir())


def read_city_se3_ego(log_dir: UPath) -> tuple[np.ndarray, np.ndarray]:
    """Read ``city_SE3_egovehicle.feather`` (at the log root).

    Returns:
        timestamps_ns: [N] uint64 sweep/pose timestamps (sorted ascending).
        T_ego_city: [N, 4, 4] float64 poses (egovehicle -> city/global frame).
    """
    cols = _read_columns(log_dir / "city_SE3_egovehicle.feather")
    order = np.argsort(cols["timestamp_ns"])
    timestamps_ns = cols["timestamp_ns"][order].astype(np.uint64)
    poses = np.stack(
        [
            se3_from_qwxyz_t(
                cols["qw"][i],
                cols["qx"][i],
                cols["qy"][i],
                cols["qz"][i],
                cols["tx_m"][i],
                cols["ty_m"][i],
                cols["tz_m"][i],
            )
            for i in order
        ]
    )
    return timestamps_ns, poses


def read_ego_se3_sensor(log_dir: UPath) -> Dict[str, np.ndarray]:
    """Read ``calibration/egovehicle_SE3_sensor.feather``.

    Returns a mapping ``sensor_name -> T_sensor_ego`` (4x4, sensor-frame point ->
    egovehicle frame).
    """
    cols = _read_columns(log_dir / "calibration" / "egovehicle_SE3_sensor.feather")
    result: Dict[str, np.ndarray] = {}
    for i, name in enumerate(cols["sensor_name"]):
        result[str(name)] = se3_from_qwxyz_t(
            cols["qw"][i],
            cols["qx"][i],
            cols["qy"][i],
            cols["qz"][i],
            cols["tx_m"][i],
            cols["ty_m"][i],
            cols["tz_m"][i],
        )
    return result


def read_intrinsics(log_dir: UPath) -> Dict[str, CameraIntrinsics]:
    """Read ``calibration/intrinsics.feather``.

    AV2 imagery is shipped undistorted, so the radial distortion coefficients
    (``k1, k2, k3``) present in the file are intentionally ignored.
    """
    cols = _read_columns(log_dir / "calibration" / "intrinsics.feather")
    result: Dict[str, CameraIntrinsics] = {}
    for i, name in enumerate(cols["sensor_name"]):
        result[str(name)] = CameraIntrinsics(
            fx=float(cols["fx_px"][i]),
            fy=float(cols["fy_px"][i]),
            cx=float(cols["cx_px"][i]),
            cy=float(cols["cy_px"][i]),
            width=int(cols["width_px"][i]),
            height=int(cols["height_px"][i]),
        )
    return result


@dataclass(frozen=True)
class LidarSweep:
    """A single AV2 lidar sweep, in the egovehicle frame.

    xyz are egomotion-compensated to ``timestamp_ns``; per-point absolute time is
    ``timestamp_ns + offset_ns``.
    """

    xyz: np.ndarray  # [N, 3] float32, egovehicle frame
    intensity: np.ndarray  # [N] float32 in [0, 1]
    laser_number: np.ndarray  # [N] uint8 in [0, 63]
    offset_ns: np.ndarray  # [N] int64, offset from sweep start
    timestamp_ns: int  # sweep reference timestamp (filename)


def read_lidar_sweep(path: UPath) -> LidarSweep:
    """Read a single lidar sweep feather file (filename is the sweep timestamp)."""
    cols = _read_columns(path)
    timestamp_ns = int(UPath(path).stem)
    return LidarSweep(
        xyz=np.stack(
            [
                cols["x"].astype(np.float32),
                cols["y"].astype(np.float32),
                cols["z"].astype(np.float32),
            ],
            axis=1,
        ),
        intensity=(cols["intensity"].astype(np.float32) / 255.0),
        laser_number=cols["laser_number"].astype(np.uint8),
        offset_ns=cols["offset_ns"].astype(np.int64),
        timestamp_ns=timestamp_ns,
    )


def read_annotations(log_dir: UPath) -> Dict[str, np.ndarray]:
    """Read ``annotations.feather`` into a column -> numpy array mapping."""
    return _read_columns(log_dir / "annotations.feather")


def list_sensor_timestamps(log_dir: UPath, sensor_kind: str, sensor_name: Optional[str] = None) -> List[int]:
    """List the sorted nanosecond timestamps available for a sensor stream.
    Args:
        sensor_kind: ``"lidar"`` or ``"cameras"``.
        sensor_name: camera name (required for ``"cameras"``; ignored for lidar).
    """
    if sensor_kind == "lidar":
        sensor_dir = log_dir / "sensors" / "lidar"
        suffix = ".feather"
    elif sensor_kind == "cameras":
        assert sensor_name is not None, "sensor_name required for cameras"
        sensor_dir = log_dir / "sensors" / "cameras" / sensor_name
        suffix = ".jpg"
    else:
        raise ValueError(f"Unknown sensor_kind: {sensor_kind}")

    if not sensor_dir.exists():
        return []

    return sorted(int(p.stem) for p in sensor_dir.iterdir() if p.name.endswith(suffix))


def assign_lidar_units(
    laser_number: np.ndarray,
    xyz_ego: np.ndarray,
    T_up_ego: np.ndarray,
    T_down_ego: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Assign each point to ``up_lidar`` or ``down_lidar``.

    Argoverse 2 distributes a single aggregated sweep from two stacked Velodyne
    VLP-32C units whose 64 beams share one ``laser_number`` range ``[0, 63]``. The
    boundary that separates the two units is not documented in the AV2 devkit, so
    we recover it from the geometry of the calibrated extrinsics.

    The two units are split into the two laser-number halves (``< 32`` and
    ``>= 32``); empirically these are the two physical sensors (at any shared
    ``offset_ns`` they point ~180 deg apart in the ego frame). To decide *which*
    half is ``up_lidar`` vs ``down_lidar`` we use per-beam elevation flatness: a
    single laser ring traces a cone of (nearly) constant elevation only in its own
    sensor frame. Mapping a half into the wrong unit's extrinsic tilts that cone
    (the two units differ in pitch/roll), inflating the per-ring elevation spread.
    We pick the labelling that minimises the summed per-ring elevation spread,
    which separates the two assignments by a wide, stable margin (~2-10x).

    Returns a mapping ``unit_name -> boolean point mask``.
    """
    lo_mask = laser_number < VLP32C_N_BEAMS
    hi_mask = ~lo_mask

    # Cost of assigning lo->up_unit and hi->down_unit (assignment A) vs swapped (B).
    cost_a = _ring_elevation_spread(
        laser_number, xyz_ego, np.arange(VLP32C_N_BEAMS), T_up_ego
    ) + _ring_elevation_spread(laser_number, xyz_ego, np.arange(VLP32C_N_BEAMS, 2 * VLP32C_N_BEAMS), T_down_ego)
    cost_b = _ring_elevation_spread(
        laser_number, xyz_ego, np.arange(VLP32C_N_BEAMS), T_down_ego
    ) + _ring_elevation_spread(laser_number, xyz_ego, np.arange(VLP32C_N_BEAMS, 2 * VLP32C_N_BEAMS), T_up_ego)

    if cost_b < cost_a:
        return {"up_lidar": hi_mask, "down_lidar": lo_mask}
    return {"up_lidar": lo_mask, "down_lidar": hi_mask}


def _ring_elevation_spread(
    laser_number: np.ndarray,
    xyz_ego: np.ndarray,
    beams: np.ndarray,
    T_unit_ego: np.ndarray,
) -> float:
    """Mean per-beam elevation standard deviation when ``beams`` are mapped to a unit.

    In the correct sensor frame each laser ring has near-constant elevation across
    azimuth, so a tight per-ring elevation distribution indicates the correct
    extrinsic. Returns the mean per-ring elevation std in degrees.
    """
    T_ego_unit = np.linalg.inv(T_unit_ego)
    pts = (T_ego_unit[:3, :3] @ xyz_ego.T).T + T_ego_unit[:3, 3]
    dist = np.linalg.norm(pts, axis=1)

    spreads: List[float] = []
    for beam in beams:
        ring = (laser_number == beam) & (dist > 2.0)
        if int(ring.sum()) < 10:
            continue
        elev = np.degrees(np.arcsin(np.clip(pts[ring, 2] / dist[ring], -1.0, 1.0)))
        spreads.append(float(np.std(elev)))

    return float(np.mean(spreads)) if spreads else float("inf")

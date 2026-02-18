# Copyright 2019 The Waymo Open Dataset Authors.
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
#
# ========================================================================
#
# Modifications Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import zlib

from dataclasses import dataclass
from typing import Optional

import numpy as np

from scipy.linalg import expm

from tools.data_converter.waymo.deps import dataset_pb2, tf


def _decode_varint(buf: bytes, pos: int) -> tuple[int, int]:
    """Decode a protobuf varint starting at `pos`, returning (value, new_pos)."""
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return result, pos
        shift += 7


@dataclass
class _ParsedMatrix:
    """Lightweight container holding the result of fast-path protobuf matrix parsing."""

    data: np.ndarray
    dims: list[int]


def _decode_zigzag(n: int) -> int:
    """Decode a ZigZag-encoded signed integer (used by protobuf sint32/sint64)."""
    return (n >> 1) ^ -(n & 1)


def _parse_packed_varints_int32(buf: bytes, start: int, end: int) -> list[int]:
    """Decode a sequence of packed varint-encoded int32 values.

    Protobuf encodes negative int32 as 10-byte sign-extended 64-bit varints,
    so we must mask to 32 bits and reinterpret as signed.
    """
    values: list[int] = []
    pos = start
    while pos < end:
        val, pos = _decode_varint(buf, pos)
        val = val & 0xFFFFFFFF
        if val > 0x7FFFFFFF:
            val -= 0x100000000
        values.append(val)
    return values


def _parse_matrix_proto(raw_compressed: bytes, dtype: type) -> _ParsedMatrix:
    """Fast-path parser for Waymo's MatrixFloat / MatrixInt32 protos.

    Parses the protobuf wire format directly to locate the packed data field
    and extracts it via np.frombuffer (for floats) or varint decoding (for int32),
    avoiding the slow Python protobuf repeated-field construction that dominates
    the cost of ParseFromString for large matrices (~1M+ elements).

    Wire layout (from waymo_open_dataset/dataset.proto):
      MatrixFloat { repeated float data = 1 [packed=true]; optional MatrixShape shape = 2; }
      MatrixInt32 { repeated int32 data = 1 [packed=true]; optional MatrixShape shape = 2; }
      MatrixShape { repeated int32 dims = 1; }

    Note: packed float fields are raw IEEE 754 bytes (4 bytes each), while packed
    int32 fields use varint encoding (variable-length per element).
    """
    buf = zlib.decompress(raw_compressed)
    pos = 0
    end = len(buf)
    data_start: Optional[int] = None
    data_end: Optional[int] = None
    dims: list[int] = []
    is_float = dtype == np.float32 or dtype == np.float64

    while pos < end:
        tag, pos = _decode_varint(buf, pos)
        field_number = tag >> 3
        wire_type = tag & 0x07

        if wire_type == 2:  # length-delimited
            length, pos = _decode_varint(buf, pos)
            if field_number == 1:  # data (packed floats or varints)
                data_start = pos
                data_end = pos + length
            elif field_number == 2:  # shape sub-message
                sub_end = pos + length
                while pos < sub_end:
                    sub_tag, pos = _decode_varint(buf, pos)
                    sub_field = sub_tag >> 3
                    sub_wire = sub_tag & 0x07
                    if sub_field == 1 and sub_wire == 0:
                        val, pos = _decode_varint(buf, pos)
                        dims.append(val)
                    elif sub_field == 1 and sub_wire == 2:
                        sub_len, pos = _decode_varint(buf, pos)
                        sub_data_end = pos + sub_len
                        while pos < sub_data_end:
                            val, pos = _decode_varint(buf, pos)
                            dims.append(val)
                    else:
                        if sub_wire == 0:
                            _, pos = _decode_varint(buf, pos)
                        elif sub_wire == 2:
                            sl, pos = _decode_varint(buf, pos)
                            pos += sl
                        else:
                            raise ValueError(f"Unexpected sub wire type {sub_wire}")
                continue  # pos already advanced past sub-message
            pos += length
        elif wire_type == 0:  # varint
            _, pos = _decode_varint(buf, pos)
        elif wire_type == 5:  # 32-bit
            pos += 4
        elif wire_type == 1:  # 64-bit
            pos += 8
        else:
            raise ValueError(f"Unexpected wire type {wire_type}")

    if data_start is None or data_end is None:
        return _ParsedMatrix(data=np.empty(0, dtype=dtype), dims=dims)

    data: np.ndarray
    if is_float:
        data = np.frombuffer(memoryview(buf)[data_start:data_end], dtype=dtype)
    else:
        data = np.array(_parse_packed_varints_int32(buf, data_start, data_end), dtype=dtype)

    return _ParsedMatrix(data=data, dims=dims)


## Functions are adapted from the official waymo open github page
def extrapolate_pose_based_on_velocity(
    T_SDC_global: np.ndarray, v_global: np.ndarray, w_global: np.ndarray, dt_sec: float
) -> np.ndarray:
    T_extrapolated = np.eye(4, dtype=np.float32)
    T_extrapolated[:3, :3] = expm(get_skew_symmetric(w_global) * dt_sec) @ T_SDC_global[:3, :3]
    T_extrapolated[:3, 3:4] = T_SDC_global[:3, 3:4] + v_global * dt_sec

    return T_extrapolated


def get_skew_symmetric(vec: np.ndarray) -> np.ndarray:
    skew_sym = np.zeros((3, 3), dtype=np.float32)

    skew_sym[0, 1] = -vec[2]
    skew_sym[0, 2] = vec[1]
    skew_sym[1, 0] = vec[2]
    skew_sym[1, 2] = -vec[0]
    skew_sym[2, 0] = -vec[1]
    skew_sym[2, 1] = vec[0]

    return skew_sym


### From here on the function are adopted from the waymo utils
def parse_range_image_and_segmentations(frame, laser_name, ri_index: int = 0, range_image_top_pose=None):
    """Parse range images and segmentations given a frame.

    Args:
       frame: open dataset frame proto
       laser_name: the name of the laser sensor to process
       ri_index: 0 for the first return, 1 for the second return.
       range_image_top_pose: optional pre-parsed top-lidar pose to avoid redundant decompression.

    Returns:
       range_image: parsed range image matrix.
       segmentations: Optional parsed segmentation matrix.
       range_image_top_pose: parsed top-lidar per-pixel pose matrix.
    """

    range_image: Optional[_ParsedMatrix] = None
    segmentation: Optional[_ParsedMatrix] = None
    need_top_pose = range_image_top_pose is None

    for laser in frame.lasers:
        laser_ri_return = laser.ri_return1 if ri_index == 0 else laser.ri_return2

        if need_top_pose and laser.name == dataset_pb2.LaserName.TOP:
            range_image_top_pose = _parse_matrix_proto(laser_ri_return.range_image_pose_compressed, np.float32)

        if laser.name != laser_name:
            continue

        if len(laser_ri_return.range_image_compressed) > 0:  # pylint: disable=g-explicit-length-test
            range_image = _parse_matrix_proto(laser_ri_return.range_image_compressed, np.float32)

        if len(laser_ri_return.segmentation_label_compressed) > 0:
            segmentation = _parse_matrix_proto(laser_ri_return.segmentation_label_compressed, np.int32)

    return range_image, segmentation, range_image_top_pose


@dataclass
class _PointCloudReturn:
    """Result of converting a single range image return to a point cloud."""

    points: np.ndarray
    segmentation: Optional[np.ndarray]
    timestamps: np.ndarray
    range_image_indices: np.ndarray
    inclinations: np.ndarray
    azimuths: np.ndarray


def convert_range_images_to_point_clouds(
    frame,
    laser_name: str,
    range_images: list[_ParsedMatrix],
    segmentation: Optional[_ParsedMatrix],
    range_image_top_pose: _ParsedMatrix,
    start_end_timestamps,
) -> list[_PointCloudReturn]:
    """Convert one or more range image returns to point clouds in a single batched pass.

    Batches all returns along the B dimension so that the expensive TF operations
    (polar coordinate computation, extrinsic/pixel-pose transforms) run only once.
    The per-return mask and gather are applied afterwards since each return has
    different valid pixels.

    Args:
      frame: open dataset frame.
      laser_name: the name of the laser sensor to process.
      range_images: list of parsed range image matrices (one per return).
      segmentation: optional segmentation for the first return (None for others).
      range_image_top_pose: parsed top-lidar per-pixel pose matrix.
      start_end_timestamps: timestamp bounds to approximate point timestamps from.

    Returns:
      List of _PointCloudReturn, one per input range image.
    """
    n_returns = len(range_images)

    frame_pose = tf.convert_to_tensor(value=np.reshape(np.array(frame.pose.transform), [4, 4]))

    # Build top-pose [H, W, 4, 4] tensor
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(value=range_image_top_pose.data),
        range_image_top_pose.dims,
    )
    range_image_top_pose_tensor_rotation = get_rotation_matrix(
        range_image_top_pose_tensor[..., 0],
        range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2],
    )
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = get_transform(
        range_image_top_pose_tensor_rotation, range_image_top_pose_tensor_translation
    )

    results: list[_PointCloudReturn] = []

    for c in frame.context.laser_calibrations:
        if c.name != laser_name:
            continue

        if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
            beam_inclinations = compute_inclination(
                tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                height=range_images[0].dims[0],
            )
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        # Stack all returns along B dimension: [n_returns, H, W, C]
        ri_tensors = [tf.reshape(tf.convert_to_tensor(value=ri.data), ri.dims) for ri in range_images]
        # Range channel only: [n_returns, H, W]
        range_batch = tf.stack([t[..., 0] for t in ri_tensors], axis=0)

        # Tile shared parameters to match batch size
        extrinsic_batch = tf.tile(tf.expand_dims(extrinsic, axis=0), [n_returns, 1, 1])
        inclination_batch = tf.tile(
            tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
            [n_returns, 1],
        )

        pixel_pose_batch = None
        frame_pose_batch = None
        if c.name == dataset_pb2.LaserName.TOP:
            pixel_pose_batch = tf.tile(
                tf.expand_dims(range_image_top_pose_tensor, axis=0),
                [n_returns, 1, 1, 1, 1],
            )
            frame_pose_batch = tf.tile(
                tf.expand_dims(frame_pose, axis=0),
                [n_returns, 1, 1],
            )

        # Single batched call for all returns
        range_image_cartesian, range_image_timestamps, azimuths = extract_point_cloud_from_range_image(
            range_batch,
            extrinsic_batch,
            inclination_batch,
            start_end_timestamps,
            pixel_pose=pixel_pose_batch,
            frame_pose=frame_pose_batch,
        )

        # Timestamps and azimuths are geometry-only (independent of range values),
        # so they're identical across returns -- use the first batch element.
        timestamps_full = range_image_timestamps[0]  # [H, W]
        azimuths_np = azimuths[0].numpy().astype(np.float32)
        inclinations_np = beam_inclinations.numpy().astype(np.float32)

        for ri_idx in range(n_returns):
            end_point_cartesian = range_image_cartesian[0][ri_idx]  # [H, W, 3]
            source_point_cartesian = range_image_cartesian[1][ri_idx]  # [H, W, 3]

            cartesian_full = tf.concat(
                [
                    source_point_cartesian,
                    end_point_cartesian,
                    ri_tensors[ri_idx][..., 1:2],  # intensity
                    ri_tensors[ri_idx][..., 2:3],  # elongation
                ],
                axis=-1,
            )  # [H, W, 8]

            range_image_mask = ri_tensors[ri_idx][..., 0] > 0
            range_image_indices_tensor = tf.compat.v1.where(range_image_mask)

            points = tf.gather_nd(cartesian_full, range_image_indices_tensor).numpy()

            seg_result = None
            if ri_idx == 0 and segmentation is not None:
                seq_tensor = tf.reshape(tf.convert_to_tensor(value=segmentation.data), segmentation.dims)
                seg_result = tf.gather_nd(seq_tensor, range_image_indices_tensor).numpy()

            ts = tf.gather_nd(timestamps_full, range_image_indices_tensor).numpy()
            ri_indices = range_image_indices_tensor.numpy().astype(np.uint32)

            results.append(
                _PointCloudReturn(
                    points=points,
                    segmentation=seg_result,
                    timestamps=ts,
                    range_image_indices=ri_indices,
                    inclinations=inclinations_np,
                    azimuths=azimuths_np,
                )
            )

    return results


def extract_point_cloud_from_range_image(
    range_image,
    extrinsic,
    inclination,
    start_end_timestamps,
    pixel_pose=None,
    frame_pose=None,
    dtype=tf.float32,
    scope=None,
):
    """Extracts point cloud from range image.

    Args:
      range_image: [B, H, W] tensor. Lidar range images.
      extrinsic: [B, 4, 4] tensor. Lidar extrinsic.
      inclination: [B, H] tensor. Inclination for each row of the range image.
        0-th entry corresponds to the 0-th row of the range image.
      start_end_timestamps [2]: timestamp bounds to approximate point timestamps from.
      pixel_pose: [B, H, W, 4, 4] tensor. If not None, it sets pose for each range
        image pixel.
      frame_pose: [B, 4, 4] tensor. This must be set when pixel_pose is set. It
        decides the vehicle frame at which the cartesian points are computed.
      dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
      scope: the name scope.

    Returns:
      range_image_cartesian: [B, H, W, 3] with {x, y, z} as inner dims in world frame.
    """
    with tf.compat.v1.name_scope(
        scope,
        "ExtractPointCloudFromRangeImage",
        [range_image, extrinsic, inclination, pixel_pose, frame_pose],
    ):
        range_image_polar, azimuths = compute_range_image_polar(range_image, extrinsic, inclination, dtype=dtype)
        range_image_cartesian, timestamps = compute_range_image_cartesian(
            range_image_polar,
            extrinsic,
            start_end_timestamps,
            pixel_pose=pixel_pose,
            frame_pose=frame_pose,
            dtype=dtype,
            return_local_coordinates=False,
        )

        return range_image_cartesian, timestamps, azimuths


def compute_inclination(inclination_range, height, scope=None):
    """Computes uniform inclination range based the given range and height.

    Args:
      inclination_range: [..., 2] tensor. Inner dims are [min inclination, max
        inclination].
      height: an integer indicates height of the range image.
      scope: the name scope.

    Returns:
      inclination: [..., height] tensor. Inclinations computed.
    """
    with tf.compat.v1.name_scope(scope, "ComputeInclination", [inclination_range]):
        diff = inclination_range[..., 1] - inclination_range[..., 0]
        inclination = (0.5 + tf.cast(tf.range(0, height), dtype=inclination_range.dtype)) / tf.cast(
            height, dtype=inclination_range.dtype
        ) * tf.expand_dims(diff, axis=-1) + inclination_range[..., 0:1]
        return inclination


def compute_range_image_polar(range_image, extrinsic, inclination, dtype=tf.float32, scope=None):
    """Computes range image polar coordinates.

    Args:
      range_image: [B, H, W] tensor. Lidar range images.
      extrinsic: [B, 4, 4] tensor. Lidar extrinsic.
      inclination: [B, H] tensor. Inclination for each row of the range image.
        0-th entry corresponds to the 0-th row of the range image.
      dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
      scope: the name scope.

    Returns:
      range_image_polar: [B, H, W, 3] polar coordinates.
    """
    # pylint: disable=unbalanced-tuple-unpacking
    _, height, width = _combined_static_and_dynamic_shape(range_image)
    range_image_dtype = range_image.dtype
    range_image = tf.cast(range_image, dtype=dtype)
    extrinsic = tf.cast(extrinsic, dtype=dtype)
    inclination = tf.cast(inclination, dtype=dtype)

    with tf.compat.v1.name_scope(scope, "ComputeRangeImagePolar", [range_image, extrinsic, inclination]):
        with tf.compat.v1.name_scope("Azimuth"):
            # [B].
            az_correction = tf.atan2(extrinsic[..., 1, 0], extrinsic[..., 0, 0])
            # [W].
            ratios = (tf.cast(tf.range(width, 0, -1), dtype=dtype) - 0.5) / tf.cast(width, dtype=dtype)
            # [B, W].
            azimuth = (ratios * 2.0 - 1.0) * np.pi - tf.expand_dims(az_correction, -1)

        # [B, H, W]
        azimuth_tile = tf.tile(azimuth[:, tf.newaxis, :], [1, height, 1])
        # [B, H, W]
        inclination_tile = tf.tile(inclination[:, :, tf.newaxis], [1, 1, width])
        range_image_polar = tf.stack([azimuth_tile, inclination_tile, range_image], axis=-1)
        return tf.cast(range_image_polar, dtype=range_image_dtype), azimuth


def compute_range_image_cartesian(
    range_image_polar,
    extrinsic,
    start_end_timestamps,
    pixel_pose=None,
    frame_pose=None,
    dtype=tf.float32,
    scope=None,
    return_local_coordinates=False,
):
    """Computes range image cartesian coordinates from polar ones.

    Args:
      range_image_polar: [B, H, W, 3] float tensor. Lidar range image in polar
        coordinate in sensor frame.
      extrinsic: [B, 4, 4] float tensor. Lidar extrinsic.
      pixel_pose: [B, H, W, 4, 4] float tensor. If not None, it sets pose for each
        range image pixel.
      frame_pose: [B, 4, 4] float tensor. This must be set when pixel_pose is set.
        It decides the vehicle frame at which the cartesian points are computed.
      dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
      scope: the name scope.

    Returns:
      range_image_cartesian: [B, H, W, 3] cartesian coordinates.
    """
    range_image_polar_dtype = range_image_polar.dtype
    range_image_polar = tf.cast(range_image_polar, dtype=dtype)
    extrinsic = tf.cast(extrinsic, dtype=dtype)
    if pixel_pose is not None:
        pixel_pose = tf.cast(pixel_pose, dtype=dtype)
    if frame_pose is not None:
        frame_pose = tf.cast(frame_pose, dtype=dtype)

    with tf.compat.v1.name_scope(
        scope,
        "ComputeRangeImageCartesian",
        [range_image_polar, extrinsic, pixel_pose, frame_pose],
    ):
        azimuth, inclination, range_image_range = tf.unstack(range_image_polar, axis=-1)

        cos_azimuth = tf.cos(azimuth)
        sin_azimuth = tf.sin(azimuth)
        cos_incl = tf.cos(inclination)
        sin_incl = tf.sin(inclination)

        # [B, H, W].
        x = cos_azimuth * cos_incl * range_image_range
        y = sin_azimuth * cos_incl * range_image_range
        z = sin_incl * range_image_range

        ## Approximate point timestamps (see https://github.com/waymo-research/waymo-open-dataset/issues/619#issuecomment-1496410092)
        #  inferred from column index [B, H, W]
        nRows, nCols = azimuth.shape[1:]
        column_relative_times = tf.reshape(tf.range(nCols) / (nCols - 1), (-1, 1, nCols))
        relative_times = tf.repeat(column_relative_times, nRows, axis=1)
        # *clockwise* spinning lidar lidar (largest azimuth are measured first)
        range_image_timestamps = start_end_timestamps[0] + tf.cast(
            relative_times * (start_end_timestamps[1] - start_end_timestamps[0]),
            dtype=start_end_timestamps[0].dtype,
        )

        # [B, H, W, 3]
        range_image_points = tf.stack([x, y, z], -1)
        range_image_center = tf.zeros_like(range_image_points)

        # [B, 3, 3]
        rotation = extrinsic[..., 0:3, 0:3]
        # translation [B, 1, 3]
        translation = tf.expand_dims(tf.expand_dims(extrinsic[..., 0:3, 3], 1), 1)

        # To vehicle frame.
        # [B, H, W, 3]
        range_image_points = tf.einsum("bkr,bijr->bijk", rotation, range_image_points) + translation

        range_image_center = tf.einsum("bkr,bijr->bijk", rotation, range_image_center) + translation
        if pixel_pose is not None:
            # To global frame.
            # [B, H, W, 3, 3]
            pixel_pose_rotation = pixel_pose[..., 0:3, 0:3]
            # [B, H, W, 3]
            pixel_pose_translation = pixel_pose[..., 0:3, 3]
            # [B, H, W, 3]
            range_image_points = (
                tf.einsum("bhwij,bhwj->bhwi", pixel_pose_rotation, range_image_points) + pixel_pose_translation
            )

            range_image_center = (
                tf.einsum("bhwij,bhwj->bhwi", pixel_pose_rotation, range_image_center) + pixel_pose_translation
            )

            if frame_pose is None:
                raise ValueError("frame_pose must be set when pixel_pose is set.")

            if return_local_coordinates:
                # To vehicle frame corresponding to the given frame_pose
                # [B, 4, 4]
                world_to_vehicle = tf.linalg.inv(frame_pose)
                world_to_vehicle_rotation = world_to_vehicle[:, 0:3, 0:3]
                world_to_vehicle_translation = world_to_vehicle[:, 0:3, 3]
                # [B, H, W, 3]
                range_image_points = (
                    tf.einsum("bij,bhwj->bhwi", world_to_vehicle_rotation, range_image_points)
                    + world_to_vehicle_translation[:, tf.newaxis, tf.newaxis, :]
                )
                range_image_center = (
                    tf.einsum("bij,bhwj->bhwi", world_to_vehicle_rotation, range_image_center)
                    + world_to_vehicle_translation[:, tf.newaxis, tf.newaxis, :]
                )

        range_image_points = tf.cast(range_image_points, dtype=range_image_polar_dtype)
        range_image_center = tf.cast(range_image_center, dtype=range_image_polar_dtype)

        return (range_image_points, range_image_center), range_image_timestamps


def get_rotation_matrix(roll, pitch, yaw, name=None):
    """Gets a rotation matrix given roll, pitch, yaw.

    roll-pitch-yaw is z-y'-x'' intrinsic rotation which means we need to apply
    x(roll) rotation first, then y(pitch) rotation, then z(yaw) rotation.

    https://en.wikipedia.org/wiki/Euler_angles
    http://planning.cs.uiuc.edu/node102.html

    Args:
      roll : x-rotation in radians.
      pitch: y-rotation in radians. The shape must be the same as roll.
      yaw: z-rotation in radians. The shape must be the same as roll.
      name: the op name.

    Returns:
      A rotation tensor with the same data type of the input. Its shape is
        [input_shape_of_yaw, 3 ,3].
    """
    with tf.compat.v1.name_scope(name, "GetRotationMatrix", [yaw, pitch, roll]):
        cos_roll = tf.cos(roll)
        sin_roll = tf.sin(roll)
        cos_yaw = tf.cos(yaw)
        sin_yaw = tf.sin(yaw)
        cos_pitch = tf.cos(pitch)
        sin_pitch = tf.sin(pitch)

        ones = tf.ones_like(yaw)
        zeros = tf.zeros_like(yaw)

        r_roll = tf.stack(
            [
                tf.stack([ones, zeros, zeros], axis=-1),
                tf.stack([zeros, cos_roll, -1.0 * sin_roll], axis=-1),
                tf.stack([zeros, sin_roll, cos_roll], axis=-1),
            ],
            axis=-2,
        )
        r_pitch = tf.stack(
            [
                tf.stack([cos_pitch, zeros, sin_pitch], axis=-1),
                tf.stack([zeros, ones, zeros], axis=-1),
                tf.stack([-1.0 * sin_pitch, zeros, cos_pitch], axis=-1),
            ],
            axis=-2,
        )
        r_yaw = tf.stack(
            [
                tf.stack([cos_yaw, -1.0 * sin_yaw, zeros], axis=-1),
                tf.stack([sin_yaw, cos_yaw, zeros], axis=-1),
                tf.stack([zeros, zeros, ones], axis=-1),
            ],
            axis=-2,
        )

        return tf.matmul(r_yaw, tf.matmul(r_pitch, r_roll))


def get_transform(rotation, translation):
    """Combines NxN rotation and Nx1 translation to (N+1)x(N+1) transform.

    Args:
      rotation: [..., N, N] rotation tensor.
      translation: [..., N] translation tensor. This must have the same type as
        rotation.

    Returns:
      transform: [..., (N+1), (N+1)] transform tensor. This has the same type as
        rotation.
    """
    with tf.name_scope("GetTransform"):
        # [..., N, 1]
        translation_n_1 = translation[..., tf.newaxis]
        # [..., N, N+1]
        transform = tf.concat([rotation, translation_n_1], axis=-1)
        # [..., N]
        last_row = tf.zeros_like(translation)
        # [..., N+1]
        last_row = tf.concat([last_row, tf.ones_like(last_row[..., 0:1])], axis=-1)
        # [..., N+1, N+1]
        transform = tf.concat([transform, last_row[..., tf.newaxis, :]], axis=-2)
        return transform


def _combined_static_and_dynamic_shape(tensor):
    """Returns a list containing static and dynamic values for the dimensions.

    Returns a list of static and dynamic values for shape dimensions. This is
    useful to preserve static shapes when available in reshape operation.

    Args:
      tensor: A tensor of any type.

    Returns:
      A list of size tensor.shape.ndims containing integers or a scalar tensor.
    """
    static_tensor_shape = tensor.shape.as_list()
    dynamic_tensor_shape = tf.shape(input=tensor)
    combined_shape = []
    for index, dim in enumerate(static_tensor_shape):
        if dim is not None:
            combined_shape.append(dim)
        else:
            combined_shape.append(dynamic_tensor_shape[index])
    return combined_shape

# Argoverse 2 to NCore V4 Converter

Converts [Argoverse 2](https://www.argoverse.org/av2.html) Sensor Dataset logs to
NCore V4 format.

The converter reads the Argoverse 2 on-disk Apache Feather files directly with
`pyarrow`, deliberately avoiding the heavy `av2` devkit (which pulls in torch,
kornia, numba, polars and PyAV). The only additional dependency is
`pyquaternion` for scalar-first quaternion conversion.

## Requirements

- Argoverse 2 Sensor Dataset downloaded locally, organised as
  `{root}/{split}/{log_id}/...`
- Python packages: `pyarrow`, `pyquaternion`

## Usage

```bash
bazel run //tools/data_converter/argoverse2 -- \
    --root-dir /path/to/argoverse2/sensor \
    --output-dir /path/to/output \
    argoverse2-v4 \
    --split val
```

### Convert a single log

```bash
bazel run //tools/data_converter/argoverse2 -- \
    --root-dir /path/to/argoverse2/sensor \
    --output-dir /path/to/output \
    argoverse2-v4 \
    --split val \
    --log-id 02678d04-cc9f-3148-9f95-1ba66347dff9
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--split` | val | Split directory under `--root-dir` (train, val, test) |
| `--log-id` | None | Filter to a single log by ID |
| `--store-type` | itar | Output store format (itar or directory) |
| `--profile` | separate-sensors | Component group assignment profile |
| `--sequence-meta/--no-sequence-meta` | enabled | Generate sequence meta JSON |

## Sensor Assumptions

- **Cameras**: 9 global-shutter cameras (7 ring + 2 stereo). AV2 imagery is shipped
  already undistorted, so the stored model is a pinhole (`ShutterType.GLOBAL`) with
  all distortion coefficients zero. The `k1, k2, k3` coefficients in
  `intrinsics.feather` are intentionally ignored.
- **Lidar**: two stacked Velodyne VLP-32C units (`up_lidar` / `down_lidar`, 10 Hz).
  The source sweep is egomotion-compensated to the sweep reference timestamp and
  expressed in the egovehicle frame. Real per-point timestamps are available via
  `offset_ns`. Each unit is stored separately with its own extrinsic. Points are
  mapped into each unit's sensor frame and decompensated using the real per-point
  timestamps so the stored directions are raw per-point-time measurements. Because
  the extrinsic is static, this is independent of whether AV2 applied ego-motion
  before or after the sensor transform.
  - No structured lidar model is stored in this version (`model_element=None`); a
    derived VLP-32C structured model can be added as a follow-up.
  - The `laser_number` to up/down unit split is not documented by AV2. The two
    units occupy the two laser-number halves (`< 32` and `>= 32`); the unit *label*
    is recovered from extrinsic geometry by per-beam elevation flatness (a laser
    ring traces a constant-elevation cone only in its own sensor frame, so the
    wrong extrinsic tilts the cone and inflates the per-ring elevation spread). The
    decision is made once per log and is stable with a wide (~2-10x) margin.
- **Radar**: AV2 has no radar.
- **Cuboid annotations**: native egovehicle frame, stored against the `rig` frame
  at the sweep timestamp (lossless, no transform). `track_uuid` is used as track ID.

## Coordinate frames

The first ego pose's `city_SE3_egovehicle` is stored as the static
`world -> world_global` pose, so `world_global` is the AV2 city frame. All absolute
city coordinates remain recoverable for later alignment with the AV2 HD map (which
the converter does not export).

## Testing

```bash
AV2_DIR=/path/to/argoverse2/sensor AV2_SPLIT=val \
    bazel test //tools/data_converter/argoverse2:pytest_converter
```

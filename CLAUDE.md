# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 app.py              # curses menu (entry point for everything)
pytest                      # run all tests
pytest tests/test_geometry.py::GeometryTests::test_direction_from_points  # single test
```

No build step. OpenCV and NumPy are imported with `try/except ModuleNotFoundError` in modules that touch hardware, so importing them on a machine without `opencv-contrib-python` succeeds — only calling their functions raises.

## Architecture

End-to-end stereo pose pipeline is **implemented and working** (RMS estéreo ≈ 2.17 px on the current calibration). Gestures, homography, and `select_points` are still stubs flagged `pending=True` in `app.py`.

### Pipeline (left + right camera → tool pose)

```
CameraSource (vision/camera.py)
  → StereoCalibration.rectify_pair       (vision/stereo.py)
  → detect_marker(A) / detect_marker(B)  (vision/color_markers.py)
  → StereoCalibration.triangulate_point   → 3D in cm
  → RodPoseEstimator.estimate_from_markers (pose/rod_pose.py)
  → build_payload  → save_last_pose       (app_state.py)
```

`main_pose.run_pose_estimation` runs this loop; `preview_marker_detection` runs only the detection half for HSV tuning.

### Where things live (and don't)

- **`pose/`** — pure math, no OpenCV. Easy to unit-test. Don't import `cv2` here.
- **`vision/`** — everything that touches frames or calibration data.
- **`calibration/`** — chessboard capture + intrinsics + stereo.
- **`gestures/`** — MediaPipe wrappers (still stubs).
- **`robot/`** — handoff surface only. No kinematics, no control. The partner project consumes the JSON payload from `state/last_pose.json`.

### Pose payload (what RoboDK consumes)

`RodPoseEstimator.build_payload` produces:

- `position_cm` = **tool tip** (not the midpoint). `position_reference: "tool_tip"`.
- `tip_position_cm` = same as position (the estimator applies `tip_offset_cm` along the A→B direction, plus y/z offsets).
- `direction` = unit vector A→B.
- `orientation` = `{format: "direction_vector", value: [...]}` — placeholder; matrix/quaternion is pending.
- `markers.center_3d_cm` / `a_3d_cm` / `b_3d_cm` / pixel coords on each camera.
- `marker_distance_cm` (measured) vs `expected_marker_distance_cm` (from config) — useful as a sanity check overlay.
- `confidence` = min across all four marker detections (left A, right A, left B, right B).
- `frame: "left_camera_rectified"` — coordinates are in the rectified left camera frame.

### Calibration — non-obvious bits

`calibration/camera_calibration.py`:

- Expects matched pairs `data/calibration_images/left_NN.png` + `right_NN.png`. Tries multiple pattern sizes (configured, transposed, 9×6, 6×9) and picks whichever yields the most detected pairs. Pattern size is **inner corners**, not squares (an 8×8-square board has 7×7 corners).
- **Corner-order alignment matters a lot on symmetric boards (7×7).** `_corner_order_candidates` enumerates id/flip_lr/flip_ud/rot180, plus the four transposed variants when columns == rows, and picks the permutation that minimizes mean point distance against the left camera. Without this, OpenCV may number corners from opposite extremes in left vs right, silently breaking stereo correspondences while still producing valid-looking individual calibrations. `reordered_pairs` in `state/calibration_info.json` lists which pairs were re-permuted.
- Uses `CALIB_FIX_INTRINSIC` for `stereoCalibrate` — individual intrinsics must be solid before stereo converges.
- After `stereoCalibrate`, runs `cv.stereoRectify(..., flags=CALIB_ZERO_DISPARITY, alpha=0)` and saves `R_left_rect`, `R_right_rect`, `P_left`, `P_right`, `Q`, ROIs into `stereo.npz`. `vision/stereo.py` rebuilds the `initUndistortRectifyMap` arrays on load — the rectification maps themselves are not serialized.

### Color detection — non-obvious bits

`vision/color_markers.threshold_color` handles the **OpenCV HSV red wraparound**: if `lower[0] > upper[0]`, it splits into two `inRange` calls (`[lower_h..179]` and `[0..upper_h]`) and ORs them. Anything red-ish in config relies on this.

Marker area is bounded both ways: `MIN_MARKER_AREA_PX = 120`, `MAX_MARKER_AREA_RATIO = 3%` of frame area. This filters out both noise and large solid-color regions (background walls, clothes) — the latter is tested in `test_color_markers.test_ignores_green_regions_too_large_to_be_a_marker`.

### Config / persistence

**`app_state.py` is the authority.** It owns `DEFAULT_CONFIG`, `load_config()`, `save_config()`, and the `save_last_*`/`append_*` helpers that write to `state/`. New modules needing config import from here — do not re-read `state/config.json` directly.

`app.py` duplicates `DEFAULT_CONFIG` because the menu must work even when downstream modules are stubs. **When adding new config keys, update both `DEFAULT_CONFIG` dicts** (in `app.py` and `app_state.py`). `_merge_defaults()` ensures old `state/config.json` files keep working when new keys appear.

Current real config in use: green/pink markers on a pen (not the red/blue bisturí placeholder), `marker_distance_cm: 8.0`, `tip_offset_cm: [-7.5, 0, 0]` (tip is opposite the A→B direction), and **cameras swapped: left=1, right=0**.

## Conventions

- Python 3, 4-space indent, `from __future__ import annotations`, `pathlib.Path`, dataclasses for records, type hints on public helpers.
- User-facing strings, comments, and docstrings are in **Spanish**. Preserve this when editing those files — code identifiers stay in English.
- Pure math lives in `pose/`; anything touching OpenCV/MediaPipe/hardware lives in `vision/`, `calibration/`, or `gestures/`. Don't cross this line — it keeps `pose/` unit-testable without a camera.
- Configured file paths (`config.persistence.*`, `config.calibration.*`) are resolved against `BASE_DIR` (project root) via local `_project_path()` helpers — follow that pattern.
- `.gitignore` excludes `calibration/*.npy`, `calibration/*.npz`, `state/logs/*.jsonl`, and `data/recordings/*` — local outputs, never commit.

## Scope boundary

This project produces data only. Robot kinematics, trajectory planning, and RoboDK simulation are explicitly out of scope (done by a partner project). `robot/` is a handoff surface for the JSON payload — do not add control logic there.

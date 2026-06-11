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

End-to-end stereo pose pipeline is **implemented and working** (RMS estéreo ≈ 2.17 px on the current calibration). Gesture control is **implemented** (MediaPipe Tasks Hand Landmarker → `open_hand`/`closed_fist`/`two_fingers`/`three_fingers` → robot stop/continue/pause/go-to-fixed-pose). The free-hand gesture only counts when it is away from the tool markers (the hand holding the tool is ignored) and is held for `confirm_frames`. Fixed poses live in `config.gestures.poses` as joint vectors in degrees (`home` = arm straight up). Homography and `select_points` are still stubs flagged `pending=True` in `app.py`.

### Pipeline (left + right camera → tool pose)

```
CameraSource (vision/camera.py)
  → read_stereo_pair (grab both, then retrieve both)
  → StereoCalibration.rectify_pair       (vision/stereo.py)
  → detect_marker(A/B/C)                 (vision/color_markers.py)
  → epipolar validation (default <= 4 px)
  → StereoCalibration.triangulate_point   → 3D in cm
  → RodPoseEstimator.estimate_from_markers (pose/rod_pose.py)
  → build_payload  → save_last_pose       (app_state.py)
```

`main_pose.run_pose_estimation` runs this loop; `preview_marker_detection` runs only the detection half for HSV tuning.

The per-frame compute lives in `main_pose.StereoPoseProcessor.process()` (pure compute → `PoseFrameResult`); drawing is `draw_pose_frame()`. Both the sequential loop (`run_pose_estimation`) and the multithreaded app (`main_app3d`) share them, so the vision logic has one source of truth.

### Augmented reality (2.3) — `augmented_reality.AugmentedRobot` / `run_pose_with_ar`

Projects a **real-scale virtual UR5 onto the left camera image**, anchored at the captured world origin, with position-only IK so its TCP follows the physical tool ("grabs" it on screen). Works in the **raw chessboard PnP frame** (the only one geometrically registered with the image) — it deliberately **ignores `world_flip_z`/`world_scale`**, which only exist for the Swift visor. K is `stereo.p_left[:3,:3]` with zero distortion (same K `solvePnP` used), so `cv2.projectPoints` is exact. `fkine_all` returns spurious origin frames (indices 9–11 on UR5); `_joint_pixels` filters them and forces the real `fkine` TCP as the last point.

### Multithreaded app (2.5) — `main_app3d.run_app_3d`

Three threads talking through single-slot latest-value mailboxes (`_LatestSlot`): a **vision** worker (capture+rectify+detect+triangulate+pose), a **gesture** worker (MediaPipe + `GestureController.update`), and the **main** thread (UR5 viewer + OpenCV windows). GUI (OpenCV/Swift) stays on the main thread — required on macOS. Real parallelism because cv2 and MediaPipe release the GIL. `GestureController` splits `update()` (analysis, runs in the gesture thread) from `draw()` (overlay, runs in main) so no frame is shared mutably; the main thread reads `controller.active`/`last_command` to drive following and fixed-pose moves.

### Where things live (and don't)

- **`pose/`** — pure math, no OpenCV. Easy to unit-test. Don't import `cv2` here.
- **`vision/`** — everything that touches frames or calibration data.
- **`calibration/`** — chessboard capture + intrinsics + stereo.
- **`gestures/`** — MediaPipe Hand Landmarker wrapper (`hand_detector.py`) + pure finger-counting classifier and `GestureController` (`gesture_classifier.py`). The classifier logic is OpenCV-free and unit-tested with synthetic landmarks. Needs `gestures/models/hand_landmarker.task` (auto-downloaded on first use, git-ignored).
- **`robot/`** — handoff surface only. No kinematics, no control. The partner project consumes the JSON payload from `state/last_pose.json`.

### Pose payload (what RoboDK consumes)

The current tool uses three markers: A=pink, B=green, C=yellow. Its local
coordinates are A=(0,0,0), B=(16.5,0,0), C=(5.8,5.8,0), in centimetres.

`RodPoseEstimator.build_payload` produces:

- `position_cm` = **marker A / tool tip**. `position_reference: "marker_a_tool_tip"`.
- `tip_position_cm` = same as position with the current zero local offset.
- `direction` = unit vector A→B.
- `orientation` = a 3x3 rotation matrix whose columns are tool X/Y/Z.
- `markers.center_3d_cm` / `a_3d_cm` / `b_3d_cm` / `c_3d_cm` plus pixel coordinates from both cameras.
- `marker_distances_cm` validates AB, AC and BC against the configured geometry.
- `epipolar_errors_px` records the rectified vertical mismatch for A/B/C.
- Only `VALID` poses are persisted or sent to the UR5 viewer.
- `confidence` = min across all six marker detections.
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

Current real config in use: A=pink TCP, B=green at 16.5 cm, C=yellow at
local `(5.8,5.8,0)`, `tip_offset_cm: [0,0,0]`, and **cameras swapped:
left=1, right=0**.

## Conventions

- Python 3, 4-space indent, `from __future__ import annotations`, `pathlib.Path`, dataclasses for records, type hints on public helpers.
- User-facing strings, comments, and docstrings are in **Spanish**. Preserve this when editing those files — code identifiers stay in English.
- Pure math lives in `pose/`; anything touching OpenCV/MediaPipe/hardware lives in `vision/`, `calibration/`, or `gestures/`. Don't cross this line — it keeps `pose/` unit-testable without a camera.
- Configured file paths (`config.persistence.*`, `config.calibration.*`) are resolved against `BASE_DIR` (project root) via local `_project_path()` helpers — follow that pattern.
- `.gitignore` excludes `calibration/*.npy`, `calibration/*.npz`, `state/logs/*.jsonl`, and `data/recordings/*` — local outputs, never commit.

## Scope boundary

This project produces data only. Robot kinematics, trajectory planning, and RoboDK simulation are explicitly out of scope (done by a partner project). `robot/` is a handoff surface for the JSON payload — do not add control logic there.

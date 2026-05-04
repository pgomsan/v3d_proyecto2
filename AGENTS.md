# Repository Guidelines

## Project Structure & Module Organization

This is a Python vision project for estimating a tool pose from color markers and handling gesture commands. Root entry points include `app.py` for the curses menu, `main_pose.py` for pose workflows, and `main_gestures.py` for gesture workflows. Shared persistence helpers live in `app_state.py`.

Core packages are organized by responsibility: `vision/` handles cameras, marker detection, and frame debugging; `pose/` contains pure geometry and pose helpers; `gestures/` wraps hand detection and classification; `calibration/` contains camera and plane calibration scripts; and `robot/` exposes the handoff surface for RoboDK integration. Persistent JSON state is in `state/`, generated recordings and samples belong under `data/`, unit tests are in `tests/`, and design notes are in `docs/`.

## Build, Test, and Development Commands

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run the application menu with `python3 app.py`. Run tests with `pytest` or `python3 -m unittest`. There is no separate build step.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation. Keep imports explicit and prefer the existing patterns: `from __future__ import annotations`, `pathlib.Path` for paths, dataclasses for simple records, and type hints for public helpers. Use `snake_case` for functions and variables, `PascalCase` for classes, and uppercase names for module constants such as `STATE_DIR`.

Keep OpenCV and hardware-dependent code in `vision/` or `calibration/`. Keep pure math in `pose/` so it remains easy to test. Existing user-facing labels and comments are mostly Spanish; preserve that style when extending those files.

## Testing Guidelines

Tests use Python's `unittest` style and are run by `pytest`. Name files `tests/test_*.py` and test methods `test_*`. Favor deterministic tests for pure functions in `pose/` and mock or isolate camera, MediaPipe, and filesystem interactions. Add regression tests when changing geometry, state serialization, or command mapping behavior.

## Commit & Pull Request Guidelines

The current history uses short descriptive messages, including Spanish summaries such as `cambios interfaz`. Keep commits concise and scoped, for example `mejora deteccion de marcas` or `add geometry tests`.

Pull requests should include a brief summary, the commands run, and any hardware assumptions such as camera indices, calibration files, or recorded data used. Include screenshots or terminal output when changing the curses menu, camera previews, or debug visualizations. Note any updates to tracked files in `state/` so reviewers can distinguish intentional configuration changes from local machine state.

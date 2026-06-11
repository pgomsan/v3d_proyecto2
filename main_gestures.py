from __future__ import annotations

from gestures.gesture_classifier import (
    preview_gesture_detection as _preview,
    run_gesture_detection as _run,
)


def preview_gesture_detection() -> None:
    """Previsualiza gestos sin guardar comandos persistentes."""
    _preview()


def run_gesture_detection() -> None:
    """Ejecuta gestos y guarda comandos en `state/`."""
    _run()

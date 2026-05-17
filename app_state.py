from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
STATE_DIR = BASE_DIR / "state"
LOGS_DIR = STATE_DIR / "logs"
CONFIG_PATH = STATE_DIR / "config.json"
CALIBRATION_INFO_PATH = STATE_DIR / "calibration_info.json"
LAST_POSE_PATH = STATE_DIR / "last_pose.json"
LAST_GESTURE_PATH = STATE_DIR / "last_gesture.json"
POSE_LOG_PATH = LOGS_DIR / "poses.jsonl"
GESTURE_LOG_PATH = LOGS_DIR / "gestures.jsonl"

DEFAULT_CONFIG: dict[str, Any] = {
    "project_title": "Pose de herramienta con marcas de color",
    "authors": [],
    "camera_index": 0,
    "camera": {
        "width": 1280,
        "height": 720,
        "fps": 30,
    },
    "cameras": {
        "left": {
            "index": 0,
            "name": "left",
            "width": 1280,
            "height": 720,
            "fps": 30,
        },
        "right": {
            "index": 1,
            "name": "right",
            "width": 1280,
            "height": 720,
            "fps": 30,
        },
    },
    "tool": {
        "tool_id": "bisturi_01",
        "tool_type": "bisturi",
        "length_cm": None,
        "marker_distance_cm": 8.0,
        "tip_offset_cm": [-7.5, 0.0, 0.0],
        "marker_a_color": "green",
        "marker_b_color": "pink",
        "marker_radius_cm": None,
    },
    "color_detection": {
        "marker_a_hsv_lower": [55, 100, 60],
        "marker_a_hsv_upper": [88, 255, 255],
        "marker_b_hsv_lower": [145, 60, 80],
        "marker_b_hsv_upper": [179, 255, 255],
    },
    "calibration": {
        "chessboard_pattern_size": [7, 7],
        "chessboard_square_size": 2.1,
        "camera_matrix_left_path": "calibration/K_left.npy",
        "distortion_left_path": "calibration/dist_left.npy",
        "camera_matrix_right_path": "calibration/K_right.npy",
        "distortion_right_path": "calibration/dist_right.npy",
        "stereo_calibration_path": "calibration/stereo.npz",
        "homography_path": "calibration/H.npy",
    },
    "persistence": {
        "last_pose_path": "state/last_pose.json",
        "last_gesture_path": "state/last_gesture.json",
        "pose_log_path": "state/logs/poses.jsonl",
        "gesture_log_path": "state/logs/gestures.jsonl",
    },
    "gestures": {
        "enabled": True,
        "backend": "mediapipe",
        "command_mapping": {
            "open_hand": "stop",
            "closed_fist": "continue",
            "two_fingers": "pause",
            "unknown": "none",
        },
    },
    "robodk_handoff": {
        "enabled": True,
        "output_frame": "camera",
        "tool_id": "bisturi_01",
        "owner": "companion_project",
    },
}


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _merge_defaults(defaults: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(defaults)
    for key, value in current.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_defaults(merged[key], value)
        else:
            merged[key] = value
    return merged


def ensure_state_dirs() -> None:
    """Crea las carpetas donde se guarda estado persistente."""
    STATE_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)


def load_json(path: Path, default: Any) -> Any:
    """Carga JSON y devuelve `default` si el archivo no existe o esta corrupto."""
    if not path.exists():
        return default

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return default


def save_json(path: Path, payload: Any) -> None:
    """Guarda JSON legible en disco.

    Usalo para configuracion y ultimo estado. Para historicos, usa las
    funciones `append_*`, que escriben JSONL.
    """
    ensure_state_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_config() -> dict[str, Any]:
    """Carga configuracion mezclandola con valores por defecto.

    Esto permite que al anadir nuevas claves al proyecto no se rompa un
    `state/config.json` antiguo.
    """
    ensure_state_dirs()
    return _merge_defaults(DEFAULT_CONFIG, load_json(CONFIG_PATH, {}))


def save_config(config: dict[str, Any]) -> None:
    """Guarda la configuracion persistente del menu y de la app."""
    save_json(CONFIG_PATH, config)


def save_calibration_info(payload: dict[str, Any]) -> None:
    """Actualiza el resumen de calibracion.

    Aqui deberias guardar errores de reproyeccion, fecha, numero de imagenes y
    rutas de los `.npy/.npz` generados.
    """
    current = load_json(CALIBRATION_INFO_PATH, {})
    current.update(payload)
    current["updated_at"] = _timestamp()
    save_json(CALIBRATION_INFO_PATH, current)


def save_last_pose(payload: dict[str, Any]) -> None:
    """Guarda solo la ultima pose estimada."""
    entry = {"timestamp": _timestamp(), **payload}
    save_json(LAST_POSE_PATH, entry)


def append_pose(payload: dict[str, Any]) -> None:
    """Anade una pose al historico y actualiza `last_pose.json`."""
    ensure_state_dirs()
    entry = {"timestamp": _timestamp(), **payload}
    with POSE_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    save_json(LAST_POSE_PATH, entry)


def save_last_gesture(payload: dict[str, Any]) -> None:
    """Guarda solo el ultimo gesto/comando."""
    entry = {"timestamp": _timestamp(), **payload}
    save_json(LAST_GESTURE_PATH, entry)


def append_gesture(payload: dict[str, Any]) -> None:
    """Anade un gesto/comando al historico y actualiza `last_gesture.json`."""
    ensure_state_dirs()
    entry = {"timestamp": _timestamp(), **payload}
    with GESTURE_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    save_json(LAST_GESTURE_PATH, entry)

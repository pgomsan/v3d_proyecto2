from __future__ import annotations

import copy
import curses
import json
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

from calibration.camera_calibration import calibrate_camera
from calibration.capture_calibration_images import capture_calibration_images
from calibration.homography import compute_homography
from calibration.select_points import select_points
from calibration.world_transform import (
    capture_world_transform,
    configure_world_scale,
    rotate_world_pitch_90deg,
    rotate_world_roll_90deg,
    rotate_world_yaw_90deg,
    toggle_world_flip_z,
)
from main_gestures import preview_gesture_detection, run_gesture_detection
from main_pose import (
    preview_marker_detection,
    run_pose_and_gestures,
    run_pose_estimation,
    run_pose_with_ur5_viewer,
)
from viewer.ur5_viewer import (
    cycle_tcp_aligned_axis,
    launch_ur5_viewer,
    run_ur5_pose_follower,
    toggle_tcp_axis_flip,
)
from vision.camera import find_available_cameras, preview_camera


BASE_DIR = Path(__file__).resolve().parent
STATE_DIR = BASE_DIR / "state"
CONFIG_PATH = STATE_DIR / "config.json"
CALIBRATION_INFO_PATH = STATE_DIR / "calibration_info.json"
LAST_POSE_PATH = STATE_DIR / "last_pose.json"
LAST_GESTURE_PATH = STATE_DIR / "last_gesture.json"
POSE_LOG_PATH = STATE_DIR / "logs" / "poses.jsonl"
GESTURE_LOG_PATH = STATE_DIR / "logs" / "gestures.jsonl"

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
        "marker_distance_cm": 16.5,
        "marker_c_along_ab_cm": 5.8,
        "marker_c_offset_cm": 5.8,
        "marker_distance_tolerance_ratio": 0.15,
        "smoothing_alpha": 1.0,
        "tcp_aligned_axis": "x",
        "tcp_axis_flip": False,
        "tip_offset_cm": [0.0, 0.0, 0.0],
        "marker_a_color": "pink",
        "marker_b_color": "green",
        "marker_c_color": "yellow",
        "marker_radius_cm": None,
    },
    "color_detection": {
        "marker_a_hsv_lower": [145, 60, 80],
        "marker_a_hsv_upper": [179, 255, 255],
        "marker_b_hsv_lower": [40, 100, 60],
        "marker_b_hsv_upper": [88, 255, 255],
        "marker_c_hsv_lower": [18, 100, 100],
        "marker_c_hsv_upper": [40, 255, 255],
    },
    "tracking": {
        "max_epipolar_error_px": 25.0,
        "fps_smoothing_alpha": 0.15,
        "pose_smoothing_alpha": 0.15,
        "max_position_jump_cm": 5.0,
        "max_orientation_jump_deg": 35.0,
        "temporal_reacquire_frames": 3,
        "max_joint_speed_deg_s": 720.0,
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
    "calibration": {
        "chessboard_pattern_size": [7, 7],
        "chessboard_square_size": 2.1,
        "calibration_pair_start_index": None,
        "calibration_pair_end_index": None,
        "stereo_outlier_max_p95_px": 4.0,
        "stereo_outlier_min_pairs": 10,
        "camera_matrix_left_path": "calibration/K_left.npy",
        "distortion_left_path": "calibration/dist_left.npy",
        "camera_matrix_right_path": "calibration/K_right.npy",
        "distortion_right_path": "calibration/dist_right.npy",
        "stereo_calibration_path": "calibration/stereo.npz",
        "homography_path": "calibration/H.npy",
        "world_roll_deg": 0.0,
        "world_pitch_deg": 0.0,
        "world_yaw_deg": 0.0,
        "world_flip_z": False,
        "world_scale": 1.0,
    },
    "persistence": {
        "last_pose_path": "state/last_pose.json",
        "last_gesture_path": "state/last_gesture.json",
        "pose_log_path": "state/logs/poses.jsonl",
        "gesture_log_path": "state/logs/gestures.jsonl",
    },
}


Action = Callable[[], None]


@dataclass(frozen=True)
class MenuItem:
    label: str
    action: Action | None
    pending: bool = False


MENU_ITEMS = [
    MenuItem("Ver camaras", preview_camera),
    MenuItem("Buscar y elegir camaras", lambda: choose_cameras()),
    MenuItem("Configurar herramienta", lambda: configure_tool()),
    MenuItem("Configurar comandos de gestos", lambda: configure_gesture_commands()),
    MenuItem("Capturar imagenes de calibracion", capture_calibration_images),
    MenuItem("Calibrar camaras", calibrate_camera),
    MenuItem("Capturar origen del mundo", capture_world_transform),
    MenuItem("Rotar mundo 90 grados (eje X / roll)", rotate_world_roll_90deg),
    MenuItem("Rotar mundo 90 grados (eje Y / pitch)", rotate_world_pitch_90deg),
    MenuItem("Rotar mundo 90 grados (eje Z / yaw)", rotate_world_yaw_90deg),
    MenuItem("Invertir eje Z del mundo (arriba/abajo)", toggle_world_flip_z),
    MenuItem("Configurar escala del mundo (world_scale)", configure_world_scale),
    MenuItem("Ciclar eje alineado del TCP (x/y/z)", cycle_tcp_aligned_axis),
    MenuItem("Invertir sentido del eje TCP", toggle_tcp_axis_flip),
    MenuItem("Seleccionar 4 puntos del plano", select_points, pending=True),
    MenuItem("Calcular homografia", compute_homography, pending=True),
    MenuItem("Probar deteccion de marcas", preview_marker_detection),
    MenuItem("Ejecutar estimacion de pose", run_pose_estimation),
    MenuItem("Camaras + visor UR5 en vivo", run_pose_with_ur5_viewer),
    MenuItem("Probar visor UR5 (pose fija)", launch_ur5_viewer),
    MenuItem("Visor UR5 siguiendo herramienta", run_ur5_pose_follower),
    MenuItem("Probar deteccion de gestos", preview_gesture_detection, pending=True),
    MenuItem("Ejecutar deteccion de gestos", run_gesture_detection, pending=True),
    MenuItem("Ejecutar pose + gestos", run_pose_and_gestures, pending=True),
    MenuItem("Ver ultimo estado guardado", lambda: show_last_saved_data()),
    MenuItem("Ver configuracion", lambda: show_config()),
    MenuItem("Salir", None),
]


MEDICAL_FRAMES = [
    [
        "                                  ",
        "              ██████              ",
        "              ██████              ",
        "              ██████              ",
        "        ██████████████████        ",
        "        ██████████████████        ",
        "        ██████████████████        ",
        "              ██████              ",
        "              ██████              ",
        "              ██████              ",
        "          ░░░░░░░░░░░░░░          ",
        "                                  ",
    ],
    [
        "                                  ",
        "              ▓█████              ",
        "              ▓█████              ",
        "              ▓█████              ",
        "        ▓▓████████████████        ",
        "        ▓▓████████████████        ",
        "        ▓▓████████████████        ",
        "              ▓█████              ",
        "              ▓█████              ",
        "              ▓█████              ",
        "          ░░░░░░░░░░░░░░          ",
        "                                  ",
    ],
    [
        "                                  ",
        "              ██████              ",
        "              ██████              ",
        "              ██████              ",
        "        ██████████████████        ",
        "        ██████████████████        ",
        "        ██████████████████        ",
        "              ██████              ",
        "              ██████              ",
        "              ██████              ",
        "          ░░░░░░░░░░░░░░          ",
        "                                  ",
    ],
    [
        "                                  ",
        "              █████▓              ",
        "              █████▓              ",
        "              █████▓              ",
        "        ████████████████▓▓        ",
        "        ████████████████▓▓        ",
        "        ████████████████▓▓        ",
        "              █████▓              ",
        "              █████▓              ",
        "              █████▓              ",
        "          ░░░░░░░░░░░░░░          ",
        "                                  ",
    ],
    [
        "                                  ",
        "              ██████              ",
        "              ██████              ",
        "              ██████              ",
        "        ██████████████████        ",
        "        ██████████████████        ",
        "        ██████████████████        ",
        "              ██████              ",
        "              ██████              ",
        "              ██████              ",
        "          ░░░░░░░░░░░░░░          ",
        "                                  ",
    ],
]
MEDICAL_CANVAS_WIDTH = max(len(line) for frame in MEDICAL_FRAMES for line in frame)
MEDICAL_CANVAS_HEIGHT = max(len(frame) for frame in MEDICAL_FRAMES)


def _merge_defaults(
    defaults: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    merged = copy.deepcopy(defaults)
    for key, value in current.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_defaults(merged[key], value)
        else:
            merged[key] = value
    return merged


def ensure_state_dirs() -> None:
    STATE_DIR.mkdir(exist_ok=True)
    POSE_LOG_PATH.parent.mkdir(exist_ok=True)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return default


def save_json(path: Path, payload: Any) -> None:
    ensure_state_dirs()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_config() -> dict[str, Any]:
    ensure_state_dirs()
    current = load_json(CONFIG_PATH, {})
    return _merge_defaults(DEFAULT_CONFIG, current)


def save_config(config: dict[str, Any]) -> None:
    save_json(CONFIG_PATH, config)


def wrap_prefixed_text(prefix: str, content: str, width: int) -> list[str]:
    if width <= len(prefix) + 1:
        return [f"{prefix}{content[: max(0, width - len(prefix))]}"]

    wrapped = textwrap.wrap(
        content,
        width=max(1, width - len(prefix)),
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not wrapped:
        return [prefix.rstrip()]

    lines = [f"{prefix}{wrapped[0]}"]
    indent = " " * len(prefix)
    lines.extend(f"{indent}{line}" for line in wrapped[1:])
    return lines


def show_config() -> None:
    config = load_config()
    print(f"\nConfiguracion actual ({CONFIG_PATH}):")
    print(json.dumps(config, indent=2, ensure_ascii=False))


def show_last_saved_data() -> None:
    last_pose = load_json(LAST_POSE_PATH, {})
    last_gesture = load_json(LAST_GESTURE_PATH, {})
    calibration_info = load_json(CALIBRATION_INFO_PATH, {})

    print("\n--- Ultima pose ---")
    print(
        json.dumps(last_pose, indent=2, ensure_ascii=False)
        if last_pose
        else "No hay pose guardada."
    )

    print("\n--- Ultimo gesto/comando ---")
    print(
        json.dumps(last_gesture, indent=2, ensure_ascii=False)
        if last_gesture
        else "No hay gesto guardado."
    )

    print("\n--- Estado de calibracion ---")
    print(
        json.dumps(calibration_info, indent=2, ensure_ascii=False)
        if calibration_info
        else "No hay informacion de calibracion guardada."
    )

    print(f"\nHistorial de poses: {POSE_LOG_PATH}")
    print(f"Historial de gestos: {GESTURE_LOG_PATH}")


def _prompt_keep_text(label: str, current: Any) -> str:
    raw = input(f"{label} [{current}]: ").strip()
    return str(current) if raw == "" else raw


def _prompt_keep_int(label: str, current: int) -> int:
    raw = input(f"{label} [{current}]: ").strip()
    if raw == "":
        return current
    try:
        return int(raw)
    except ValueError:
        print("Valor no valido. Se mantiene el anterior.")
        return current


def _prompt_keep_float_or_none(label: str, current: float | None) -> float | None:
    raw = input(f"{label} [{current}]: ").strip()
    if raw == "":
        return current
    if raw.lower() in {"none", "null", "-"}:
        return None
    try:
        return float(raw)
    except ValueError:
        print("Valor no valido. Se mantiene el anterior.")
        return current


def _prompt_keep_vector(label: str, current: list[float]) -> list[float]:
    raw = input(f"{label} [{', '.join(str(value) for value in current)}]: ").strip()
    if raw == "":
        return current

    try:
        values = [float(part.strip()) for part in raw.split(",")]
    except ValueError:
        print("Vector no valido. Usa formato x,y,z. Se mantiene el anterior.")
        return current

    if len(values) != 3:
        print(
            "Vector no valido. Usa exactamente tres valores. Se mantiene el anterior."
        )
        return current
    return values


def choose_cameras() -> None:
    config = load_config()
    cameras = config.setdefault("cameras", {})
    shared_camera = config.setdefault("camera", {})
    left = cameras.setdefault("left", {"index": 0, "name": "left"})
    right = cameras.setdefault("right", {"index": 1, "name": "right"})

    print("\nBuscando camaras disponibles con OpenCV...")
    try:
        available = find_available_cameras()
    except Exception as exc:
        available = []
        print(f"No se pudo hacer busqueda automatica: {exc}")

    if available:
        print("Indices disponibles:", ", ".join(str(index) for index in available))
    else:
        print(
            "No se ha detectado ninguna camara automaticamente. Puedes escribir indices manualmente."
        )

    print("\nDeja un campo vacio para mantener el valor actual.")
    left["index"] = _prompt_keep_int(
        "Indice camara izquierda", int(left.get("index", 0))
    )
    right["index"] = _prompt_keep_int(
        "Indice camara derecha", int(right.get("index", 1))
    )
    left["name"] = "left"
    right["name"] = "right"

    width = _prompt_keep_int(
        "Ancho de captura comun",
        int(shared_camera.get("width", left.get("width", 1280))),
    )
    height = _prompt_keep_int(
        "Alto de captura comun",
        int(shared_camera.get("height", left.get("height", 720))),
    )
    fps = _prompt_keep_int(
        "FPS comun", int(shared_camera.get("fps", left.get("fps", 30)))
    )

    shared_camera.update({"width": width, "height": height, "fps": fps})
    left.update({"width": width, "height": height, "fps": fps})
    right.update({"width": width, "height": height, "fps": fps})
    config["camera_index"] = left["index"]

    save_config(config)
    print(f"Camaras guardadas: izquierda={left['index']}, derecha={right['index']}.")


def configure_tool() -> None:
    config = load_config()
    tool = config.setdefault("tool", {})

    print("\nDeja un campo vacio para mantener el valor actual.")
    tool["tool_id"] = _prompt_keep_text(
        "ID de herramienta", tool.get("tool_id", "bisturi_01")
    )
    tool["tool_type"] = _prompt_keep_text(
        "Tipo de herramienta", tool.get("tool_type", "bisturi")
    )
    tool["length_cm"] = _prompt_keep_float_or_none(
        "Longitud total en cm", tool.get("length_cm")
    )
    tool["marker_distance_cm"] = _prompt_keep_float_or_none(
        "Distancia A-B en cm",
        tool.get("marker_distance_cm"),
    )
    tool["marker_c_along_ab_cm"] = _prompt_keep_float_or_none(
        "Posicion de la rama C desde A sobre A-B en cm",
        tool.get("marker_c_along_ab_cm"),
    )
    tool["marker_c_offset_cm"] = _prompt_keep_float_or_none(
        "Longitud perpendicular de la rama C en cm",
        tool.get("marker_c_offset_cm"),
    )
    tool["marker_radius_cm"] = _prompt_keep_float_or_none(
        "Radio de marca en cm", tool.get("marker_radius_cm")
    )
    tool["tip_offset_cm"] = _prompt_keep_vector(
        "Offset hasta punta util x,y,z cm", tool.get("tip_offset_cm", [0, 0, 0])
    )
    tool["marker_a_color"] = _prompt_keep_text(
        "Color marca A", tool.get("marker_a_color", "pink")
    )
    tool["marker_b_color"] = _prompt_keep_text(
        "Color marca B", tool.get("marker_b_color", "green")
    )
    tool["marker_c_color"] = _prompt_keep_text(
        "Color marca C", tool.get("marker_c_color", "yellow")
    )

    config.setdefault("robodk_handoff", {})["tool_id"] = tool["tool_id"]
    save_config(config)
    print("Parametros de herramienta guardados.")


def configure_gesture_commands() -> None:
    config = load_config()
    gestures = config.setdefault("gestures", {})
    mapping = gestures.setdefault("command_mapping", {})

    print("\nDeja un campo vacio para mantener el valor actual.")
    mapping["open_hand"] = _prompt_keep_text(
        "Comando para open_hand", mapping.get("open_hand", "stop")
    )
    mapping["closed_fist"] = _prompt_keep_text(
        "Comando para closed_fist", mapping.get("closed_fist", "continue")
    )
    mapping["two_fingers"] = _prompt_keep_text(
        "Comando para two_fingers", mapping.get("two_fingers", "pause")
    )
    mapping["unknown"] = _prompt_keep_text(
        "Comando para unknown", mapping.get("unknown", "none")
    )
    save_config(config)
    print("Mapa de gestos guardado.")


def _safe_addstr(stdscr, row: int, col: int, text: str, attr: int = 0) -> None:
    height, width = stdscr.getmaxyx()
    if row < 0 or col < 0 or row >= height or col >= width:
        return
    clipped = text[: max(0, width - col - 1)]
    if not clipped:
        return
    try:
        stdscr.addstr(row, col, clipped, attr)
    except curses.error:
        pass


def _draw_panel(
    stdscr, panel_y: int, panel_x: int, panel_height: int, panel_width: int
) -> None:
    border_attr = curses.color_pair(5)
    for x in range(panel_x, panel_x + panel_width):
        _safe_addstr(stdscr, panel_y, x, " ", border_attr)
        _safe_addstr(stdscr, panel_y + panel_height - 1, x, " ", border_attr)
    for y in range(panel_y, panel_y + panel_height):
        _safe_addstr(stdscr, y, panel_x, " ", border_attr)
        _safe_addstr(stdscr, y, panel_x + panel_width - 1, " ", border_attr)


def _medical_char_attr(char: str) -> int:
    if char in {"█", "▄", "▀"}:
        return curses.color_pair(6) | curses.A_BOLD
    if char in {"▓", "▒"}:
        return curses.color_pair(7) | curses.A_BOLD
    if char == "░":
        return curses.color_pair(9)
    return curses.color_pair(8) | curses.A_BOLD


def _draw_medical_art(
    stdscr, row: int, col: int, frame_index: int, max_width: int
) -> int:
    frame = MEDICAL_FRAMES[frame_index % len(MEDICAL_FRAMES)]
    for y_offset in range(MEDICAL_CANVAS_HEIGHT):
        line = frame[y_offset] if y_offset < len(frame) else ""
        for x_offset, char in enumerate(line[:max_width]):
            if char == " ":
                continue
            _safe_addstr(
                stdscr, row + y_offset, col + x_offset, char, _medical_char_attr(char)
            )

    return MEDICAL_CANVAS_WIDTH


def draw_menu(stdscr, selected_idx: int, frame_index: int) -> None:
    stdscr.clear()
    height, width = stdscr.getmaxyx()
    config = load_config()
    tool = config.get("tool", {})
    gestures = config.get("gestures", {})
    mapping = gestures.get("command_mapping", {})

    title = config.get("project_title", "Pose de herramienta con marcas de color")
    authors = config.get("authors", [])
    cameras = config.get("cameras", {})
    left_camera = cameras.get("left", {})
    right_camera = cameras.get("right", {})
    author_text = ", ".join(authors) if authors else "Por definir"
    tool_text = (
        f"{tool.get('tool_id', 'bisturi_01')} ({tool.get('tool_type', 'bisturi')})"
    )
    marker_text = (
        f"A={tool.get('marker_a_color', 'pink')}, "
        f"B={tool.get('marker_b_color', 'green')}, "
        f"C={tool.get('marker_c_color', 'yellow')}"
    )
    camera_text = f"L={left_camera.get('index', 0)} / R={right_camera.get('index', 1)}"
    command_text = (
        f"open={mapping.get('open_hand', 'stop')}, "
        f"fist={mapping.get('closed_fist', 'continue')}, "
        f"two={mapping.get('two_fingers', 'pause')}"
    )

    panel_width = min(104, max(76, width - 4))
    panel_height = min(height - 2, 34)
    panel_x = max(2, (width - panel_width) // 2)
    panel_y = max(1, (height - panel_height) // 2)
    _draw_panel(stdscr, panel_y, panel_x, panel_height, panel_width)

    art_x = panel_x + 4
    art_y = panel_y + 3
    max_art_width = min(MEDICAL_CANVAS_WIDTH, panel_width - 8)
    art_width = _draw_medical_art(stdscr, art_y, art_x, frame_index, max_art_width)

    text_x = art_x + min(art_width, max_art_width) + 4
    text_width = panel_x + panel_width - 4 - text_x
    if text_width < 28:
        text_x = panel_x + 4
        text_width = panel_width - 8
        art_y = panel_y + 2
        title_y = art_y + MEDICAL_CANVAS_HEIGHT + 1
    else:
        title_y = art_y + 1

    _safe_addstr(stdscr, title_y, text_x, title, curses.color_pair(2) | curses.A_BOLD)
    meta_y = title_y + 2
    meta_lines = [
        "Usa flechas y Enter para seleccionar",
        *wrap_prefixed_text("Autores: ", author_text, max(10, text_width)),
        f"Camaras: {camera_text}",
        f"Herramienta: {tool_text}",
        f"Marcas: {marker_text}",
        f"Gestos: {command_text}",
        "RoboDK: entrega de datos, no control",
        f"Persistencia: {STATE_DIR.name}/",
    ]

    for offset, line in enumerate(meta_lines):
        _safe_addstr(
            stdscr, meta_y + offset, text_x, line[:text_width], curses.color_pair(3)
        )

    divider_y = max(art_y + MEDICAL_CANVAS_HEIGHT, meta_y + len(meta_lines)) + 1
    divider_width = panel_width - 8
    divider = ("=" * divider_width)[:divider_width]
    _safe_addstr(
        stdscr, divider_y, panel_x + 4, divider, curses.color_pair(10) | curses.A_BOLD
    )

    start_row = divider_y + 2
    visible_rows = max(1, panel_y + panel_height - 3 - start_row)
    scroll_offset = 0
    if selected_idx >= visible_rows:
        scroll_offset = selected_idx - visible_rows + 1

    visible_items = MENU_ITEMS[scroll_offset : scroll_offset + visible_rows]
    for local_idx, item in enumerate(visible_items):
        idx = scroll_offset + local_idx
        row = start_row + local_idx
        marker_x = panel_x + 4
        label_x = panel_x + 7
        item_width = panel_width - 11
        label = item.label.ljust(item_width)

        if idx == selected_idx:
            _safe_addstr(
                stdscr, row, marker_x, ">", curses.color_pair(6) | curses.A_BOLD
            )
            _safe_addstr(
                stdscr, row, label_x, label, curses.color_pair(2) | curses.A_BOLD
            )
        else:
            _safe_addstr(stdscr, row, marker_x, " ", curses.color_pair(4))
            _safe_addstr(stdscr, row, label_x, label, curses.color_pair(4))

    if scroll_offset > 0:
        _safe_addstr(
            stdscr,
            start_row - 1,
            panel_x + panel_width - 10,
            "arriba",
            curses.color_pair(3),
        )
    if scroll_offset + visible_rows < len(MENU_ITEMS):
        _safe_addstr(
            stdscr,
            panel_y + panel_height - 3,
            panel_x + panel_width - 10,
            "abajo",
            curses.color_pair(3),
        )

    footer = "[Enter] abrir   [q/Esc] salir"
    _safe_addstr(
        stdscr,
        panel_y + panel_height - 2,
        panel_x + 4,
        footer[: panel_width - 8],
        curses.color_pair(3),
    )
    stdscr.refresh()


def curses_menu(stdscr) -> str:
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    stdscr.timeout(90)

    medical_color = curses.COLOR_RED
    dark_medical_color = curses.COLOR_MAGENTA
    shadow_color = curses.COLOR_MAGENTA
    if curses.can_change_color():
        try:
            curses.init_color(20, 1000, 0, 0)
            curses.init_color(21, 520, 0, 0)
            curses.init_color(22, 260, 0, 0)
            medical_color = 20
            dark_medical_color = 21
            shadow_color = 22
        except curses.error:
            medical_color = curses.COLOR_RED
            dark_medical_color = curses.COLOR_MAGENTA
            shadow_color = curses.COLOR_MAGENTA

    curses.init_pair(2, curses.COLOR_WHITE, -1)
    curses.init_pair(3, curses.COLOR_CYAN, -1)
    curses.init_pair(4, curses.COLOR_WHITE, -1)
    curses.init_pair(5, curses.COLOR_BLUE, -1)
    curses.init_pair(6, medical_color, -1)
    curses.init_pair(7, dark_medical_color, -1)
    curses.init_pair(8, curses.COLOR_WHITE, -1)
    curses.init_pair(9, shadow_color, -1)
    curses.init_pair(10, medical_color, -1)

    selected_idx = 0
    while True:
        frame_index = int(time.monotonic() * 3) % len(MEDICAL_FRAMES)
        draw_menu(stdscr, selected_idx, frame_index)
        key = stdscr.getch()

        if key == -1:
            continue
        if key in (curses.KEY_UP, ord("k")):
            selected_idx = (selected_idx - 1) % len(MENU_ITEMS)
        elif key in (curses.KEY_DOWN, ord("j")):
            selected_idx = (selected_idx + 1) % len(MENU_ITEMS)
        elif key in (10, 13, curses.KEY_ENTER):
            return MENU_ITEMS[selected_idx].label
        elif key in (27, ord("q")):
            return "Salir"


def fallback_numeric_menu() -> str:
    print("\n=== Menu principal ===")
    for index, item in enumerate(MENU_ITEMS, start=1):
        print(f"{index}. {item.label}")

    raw = input("Selecciona una opcion: ").strip()
    try:
        selected = int(raw) - 1
    except ValueError:
        return ""

    if 0 <= selected < len(MENU_ITEMS):
        return MENU_ITEMS[selected].label
    return ""


def pause_after_action() -> None:
    if sys.stdin.isatty():
        input("\nPulsa Enter para volver al menu...")


def run_menu_action(option: str) -> bool:
    selected = next((item for item in MENU_ITEMS if item.label == option), None)
    if selected is None:
        print("Opcion no valida.")
        pause_after_action()
        return True

    if selected.label == "Salir":
        print("Saliendo.")
        return False

    if selected.action is None:
        print("Opcion sin accion asociada.")
        pause_after_action()
        return True

    print(f"\n--- {selected.label} ---")
    try:
        selected.action()
        if selected.pending:
            print(
                "Esqueleto preparado. Completa esta funcion en su modulo correspondiente."
            )
    except FileNotFoundError as exc:
        print(exc)
    except Exception as exc:
        print(f"Error ejecutando la opcion: {exc}")

    pause_after_action()
    return True


def main() -> None:
    while True:
        if sys.stdin.isatty() and sys.stdout.isatty():
            option = curses.wrapper(curses_menu)
        else:
            option = fallback_numeric_menu()

        if not run_menu_action(option):
            return


if __name__ == "__main__":
    main()

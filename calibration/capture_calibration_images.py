from __future__ import annotations

import re
from pathlib import Path

from app_state import save_calibration_info
from vision.camera import (
    CameraSource,
    load_stereo_camera_configs,
    read_stereo_pair,
)

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "data" / "calibration_images"
PAIR_NAME_RE = re.compile(r"^(left|right)_(\d+)\.png$")


def _next_pair_index(output_dir: Path) -> int:
    """Devuelve el siguiente indice libre para `left_XX.png` y `right_XX.png`."""
    max_index = -1
    if not output_dir.exists():
        return 0

    for path in output_dir.iterdir():
        match = PAIR_NAME_RE.match(path.name)
        if match is not None:
            max_index = max(max_index, int(match.group(2)))
    return max_index + 1


def _draw_capture_overlay(frame, label: str, camera_index: int, next_index: int):
    """Anade ayuda visual a una copia del frame; la imagen guardada queda limpia."""
    preview = frame.copy()
    lines = [
        f"{label} camara {camera_index}",
        f"Siguiente par: {next_index:02d}",
        "Espacio/c/Enter: guardar   q/Esc: salir",
    ]

    for row, text in enumerate(lines):
        y = 30 + row * 28
        cv2.putText(
            preview,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return preview


def _save_pair(output_dir: Path, index: int, left_frame, right_frame) -> tuple[Path, Path]:
    """Guarda un par estereo con el mismo indice para ambas camaras."""
    left_path = output_dir / f"left_{index:02d}.png"
    right_path = output_dir / f"right_{index:02d}.png"

    left_ok = cv2.imwrite(str(left_path), left_frame)
    right_ok = cv2.imwrite(str(right_path), right_frame)
    if not left_ok or not right_ok:
        left_path.unlink(missing_ok=True)
        right_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"No se pudo guardar el par de calibracion {index:02d} en {output_dir}."
        )

    return left_path, right_path


def capture_calibration_images() -> None:
    """Captura pares de imagenes para calibracion estereo.

    Carga las camaras izquierda/derecha desde `state/config.json`, muestra los
    frames en ventanas OpenCV y guarda pares sincronizados en
    `data/calibration_images/left_XX.png` y `right_XX.png`.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV no esta instalado. Instala opencv-contrib-python.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    next_index = _next_pair_index(OUTPUT_DIR)
    saved_count = 0

    left_config, right_config = load_stereo_camera_configs()
    if left_config.index == right_config.index:
        print(
            "Aviso: izquierda y derecha usan el mismo indice. "
            "Para calibracion estereo deberian ser camaras distintas."
        )

    left_camera = CameraSource(left_config)
    right_camera = CameraSource(right_config)

    print(f"Guardando pares de calibracion en: {OUTPUT_DIR}")
    print("Pulsa espacio, c o Enter en una ventana OpenCV para guardar un par.")
    print("Pulsa q o Esc para terminar.")

    try:
        left_camera.open()
        right_camera.open()

        cv2.namedWindow("Calibracion izquierda", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Calibracion derecha", cv2.WINDOW_NORMAL)

        while True:
            left_frame, right_frame = read_stereo_pair(left_camera, right_camera)

            cv2.imshow(
                "Calibracion izquierda",
                _draw_capture_overlay(
                    left_frame, "Izquierda", left_config.index, next_index
                ),
            )
            cv2.imshow(
                "Calibracion derecha",
                _draw_capture_overlay(
                    right_frame, "Derecha", right_config.index, next_index
                ),
            )

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord(" "), ord("c"), 13):
                left_path, right_path = _save_pair(
                    OUTPUT_DIR, next_index, left_frame, right_frame
                )
                print(
                    "Par guardado: "
                    f"{left_path.relative_to(BASE_DIR)} / "
                    f"{right_path.relative_to(BASE_DIR)}"
                )
                saved_count += 1
                next_index += 1
    finally:
        left_camera.release()
        right_camera.release()
        cv2.destroyAllWindows()

    save_calibration_info(
        {
            "calibration_images_dir": str(OUTPUT_DIR.relative_to(BASE_DIR)),
            "last_capture_pairs": saved_count,
            "next_capture_index": next_index,
            "left_camera_index": left_config.index,
            "right_camera_index": right_config.index,
        }
    )
    print(f"Captura finalizada. Pares guardados en esta sesion: {saved_count}.")

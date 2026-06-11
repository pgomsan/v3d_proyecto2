"""Aplicacion 3D integrada con procesamiento paralelo multihilo (apartado 2.5).

Reparte el trabajo en tres hilos que se comunican por "casilleros de ultimo
valor" (se queda solo el frame mas reciente, sin acumular retardo):

  - Hilo de VISION: captura el par estereo, rectifica, detecta las marcas,
    triangula y estima la pose de la herramienta.
  - Hilo de GESTOS: corre MediaPipe sobre la imagen izquierda y clasifica el
    gesto de la mano libre (ignora la que sostiene la herramienta).
  - Hilo PRINCIPAL: actualiza el visor UR5, dibuja los overlays y muestra las
    ventanas de OpenCV (la GUI debe vivir en el hilo principal en macOS).

El paralelismo es real porque OpenCV y MediaPipe liberan el GIL durante sus
llamadas nativas: la triangulacion y la inferencia de manos se solapan.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from app_state import load_config, save_last_pose
from gestures.gesture_classifier import GestureController
from main_pose import (
    StereoPoseProcessor,
    _build_ur5_visualizer,
    _smoothed_fps,
    draw_pose_frame,
)
from vision.camera import (
    CameraSource,
    load_stereo_camera_configs,
    read_stereo_pair,
)
from vision.frame_debug import show_debug_frame

try:
    import cv2 as cv
except ModuleNotFoundError:
    cv = None


@dataclass
class _VisionOut:
    """Salida del hilo de vision para un par estereo procesado."""

    frame_id: int
    left_debug: Any
    right_debug: Any
    left_for_gestures: Any
    tool_pixels: list[tuple[float, float]]
    pose_payload: dict[str, Any] | None
    left_index: int
    right_index: int


class _LatestSlot:
    """Casillero de un solo hueco protegido por candado.

    `put` sobrescribe; `get` devuelve siempre el ultimo elemento. Asi los
    consumidores trabajan con el frame mas reciente y no se acumula latencia
    si un hilo va mas lento que otro.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: Any = None

    def put(self, item: Any) -> None:
        with self._lock:
            self._item = item

    def get(self) -> Any:
        with self._lock:
            return self._item


def _vision_worker(
    processor: StereoPoseProcessor,
    slot: _LatestSlot,
    stop_event: threading.Event,
    fps_alpha: float,
) -> None:
    """Captura y procesa pares estereo hasta que se pide parar."""
    left_config, right_config = load_stereo_camera_configs()
    left_camera = CameraSource(left_config)
    right_camera = CameraSource(right_config)
    previous_at = time.monotonic()
    fps = 0.0
    frame_id = 0
    try:
        left_camera.open()
        right_camera.open()
        while not stop_event.is_set():
            left_frame, right_frame = read_stereo_pair(left_camera, right_camera)
            now = time.monotonic()
            fps = _smoothed_fps(fps, previous_at, now, fps_alpha)
            previous_at = now

            result = processor.process(left_frame, right_frame, fps)
            left_debug, right_debug = draw_pose_frame(result)
            frame_id += 1
            slot.put(
                _VisionOut(
                    frame_id=frame_id,
                    left_debug=left_debug,
                    right_debug=right_debug,
                    left_for_gestures=result.left_rect,
                    tool_pixels=result.tool_pixels,
                    pose_payload=result.pose_payload,
                    left_index=left_config.index,
                    right_index=right_config.index,
                )
            )
    finally:
        left_camera.release()
        right_camera.release()


def _gesture_worker(
    controller: GestureController,
    slot: _LatestSlot,
    stop_event: threading.Event,
) -> None:
    """Corre MediaPipe sobre el ultimo frame de vision y actualiza el estado."""
    last_processed = -1
    while not stop_event.is_set():
        vo = slot.get()
        if vo is None or vo.frame_id == last_processed:
            time.sleep(0.005)
            continue
        last_processed = vo.frame_id
        controller.update(vo.left_for_gestures, vo.tool_pixels)


def run_app_3d() -> None:
    """App 3D integrada multihilo: vision + gestos + visor UR5 en paralelo."""
    if cv is None:
        print("OpenCV no esta instalado. Instala opencv-contrib-python.")
        return

    config = load_config()
    gestures_config = config.get("gestures", {})
    mapping = gestures_config.get("command_mapping", {})
    try:
        exclusion_radius_ratio = float(
            gestures_config.get("exclusion_radius_ratio", 0.18)
        )
    except (TypeError, ValueError):
        exclusion_radius_ratio = 0.18
    try:
        confirm_frames = int(gestures_config.get("confirm_frames", 6))
    except (TypeError, ValueError):
        confirm_frames = 6
    poses = gestures_config.get("poses", {})

    tracking_config = config.get("tracking", {})
    try:
        fps_alpha = float(tracking_config.get("fps_smoothing_alpha", 0.15))
    except (TypeError, ValueError):
        fps_alpha = 0.15

    processor = StereoPoseProcessor(config)
    print(f"RMS estereo cargado: {processor.stereo.rms:.3f}")
    print(f"Limite epipolar: {processor.max_epipolar_error_px:.1f} px")

    visualizer = _build_ur5_visualizer(config)
    if not visualizer.launch():
        return

    # En modo multihilo el visor lo toca solo el hilo principal, asi que el
    # controlador no usa callback: el hilo principal lee su estado y reacciona.
    controller = GestureController(
        mapping,
        exclusion_radius_ratio=exclusion_radius_ratio,
        confirm_frames=confirm_frames,
    )

    slot = _LatestSlot()
    stop_event = threading.Event()
    vision_thread = threading.Thread(
        target=_vision_worker,
        args=(processor, slot, stop_event, fps_alpha),
        name="vision",
        daemon=True,
    )
    gesture_thread = threading.Thread(
        target=_gesture_worker,
        args=(controller, slot, stop_event),
        name="gestos",
        daemon=True,
    )
    vision_thread.start()
    gesture_thread.start()

    print(
        "App 3D multihilo en marcha. Gesto de mano libre: puno=seguir, "
        "mano abierta/dos dedos=parar, tres dedos=recto arriba. Pulsa q o Esc."
    )

    prev_command = controller.last_command
    last_saved_at = 0.0
    try:
        while True:
            vo = slot.get()
            if vo is not None:
                # El robot solo sigue la herramienta si el gesto lo permite.
                if vo.pose_payload is not None and controller.active:
                    visualizer.update_from_payload(vo.pose_payload)

                # Flanco de comando: si pasa a un comando de pose fija, mover.
                command = controller.last_command
                if command != prev_command:
                    pose = poses.get(command)
                    if pose is not None:
                        visualizer.move_to_named_pose(pose)
                    prev_command = command

                now = time.monotonic()
                if vo.pose_payload is not None and now - last_saved_at >= 0.5:
                    save_last_pose(vo.pose_payload)
                    last_saved_at = now

                controller.draw(vo.left_debug)
                show_debug_frame(
                    f"Pose izquierda rectificada [{vo.left_index}]", vo.left_debug
                )
                show_debug_frame(
                    f"Pose derecha rectificada [{vo.right_index}]", vo.right_debug
                )

            visualizer.step()

            key = cv.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        stop_event.set()
        vision_thread.join(timeout=2.0)
        gesture_thread.join(timeout=2.0)
        controller.close()
        visualizer.close()
        cv.destroyAllWindows()

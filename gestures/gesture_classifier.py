from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from gestures.hand_detector import HandDetector, HandLandmarks

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


# Indices de landmarks de MediaPipe Hands (21 puntos).
WRIST = 0
# Para cada dedo largo: (punta, articulacion intermedia PIP).
_LONG_FINGERS = {
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18),
}


@dataclass
class GestureResult:
    gesture: str
    command: str
    confidence: float


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def fingers_extended(points: list[tuple[float, float, float]]) -> dict[str, bool]:
    """Indica que dedos largos estan extendidos.

    Usa la distancia punta->muneca frente a articulacion->muneca, de modo que
    el criterio es invariante a la rotacion de la mano (no exige que los dedos
    apunten hacia arriba).
    """
    wrist = points[WRIST]
    estado: dict[str, bool] = {}
    for name, (tip, pip) in _LONG_FINGERS.items():
        estado[name] = _distance(points[tip], wrist) > _distance(points[pip], wrist)
    return estado


def classify_gesture(hand: HandLandmarks) -> GestureResult:
    """Clasifica una mano a partir de sus landmarks.

    - `open_hand`: cuatro dedos largos extendidos;
    - `closed_fist`: ningun dedo largo extendido;
    - `two_fingers`: indice y corazon extendidos, anular y menique cerrados;
    - `unknown`: cualquier otro patron.
    """
    if not hand.points or len(hand.points) < 21:
        return GestureResult("unknown", "none", 0.0)

    estado = fingers_extended(hand.points)
    extendidos = sum(1 for abierto in estado.values() if abierto)

    if extendidos == 0:
        gesture = "closed_fist"
    elif extendidos >= 4:
        gesture = "open_hand"
    elif (
        extendidos == 2
        and estado["index"]
        and estado["middle"]
        and not estado["ring"]
        and not estado["pinky"]
    ):
        gesture = "two_fingers"
    elif (
        extendidos == 3
        and estado["index"]
        and estado["middle"]
        and estado["ring"]
        and not estado["pinky"]
    ):
        gesture = "three_fingers"
    else:
        gesture = "unknown"

    return GestureResult(gesture, "none", hand.confidence)


def gesture_to_command(gesture: str, command_mapping: dict[str, str]) -> str:
    """Convierte un gesto en comando de alto nivel para RoboDK."""
    return command_mapping.get(gesture, command_mapping.get("unknown", "none"))


def _draw_overlay(frame: Any, result: GestureResult) -> None:
    """Dibuja el gesto y comando detectados sobre el frame."""
    if cv2 is None:
        return
    texto = f"{result.gesture} -> {result.command} ({result.confidence:.2f})"
    color = (0, 200, 0) if result.gesture != "unknown" else (0, 0, 200)
    cv2.putText(
        frame, texto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA
    )


# Unico comando que mantiene el seguimiento de la herramienta. Cualquier otro
# (stop, pause, ir a una pose...) detiene el seguimiento.
_RESUME_COMMANDS = {"continue"}


def _hand_centroid_px(
    points: list[tuple[float, float, float]], width: int, height: int
) -> tuple[float, float]:
    """Centroide en pixeles de los landmarks normalizados de una mano."""
    xs = [p[0] * width for p in points]
    ys = [p[1] * height for p in points]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


class GestureController:
    """Traduce gestos a un estado de seguimiento para el visor del robot.

    Mantiene un flag `active`: cuando esta activo el robot sigue la
    herramienta; cuando un gesto pide parar/pausar, el flag se desactiva y el
    visor conserva la ultima pose. El cambio de comando se persiste en
    `state/` para que el proyecto companero pueda leerlo.

    Para no confundir la mano que sostiene la herramienta con la mano que da
    ordenes:
      - se descarta toda mano cuyo centroide caiga dentro de la "zona muerta"
        alrededor de los marcadores de la herramienta (la que la sostiene);
      - de las manos restantes se obedece a la mas alejada de la herramienta;
      - un gesto solo cambia el estado si se mantiene `confirm_frames` frames
        seguidos (histeresis), evitando disparos por formas transitorias.
    """

    def __init__(
        self,
        command_mapping: dict[str, str],
        detector: HandDetector | None = None,
        persist: bool = True,
        exclusion_radius_ratio: float = 0.18,
        confirm_frames: int = 6,
        on_command: Callable[[str], None] | None = None,
    ) -> None:
        self.command_mapping = command_mapping
        self.detector = (
            detector if detector is not None else HandDetector(max_num_hands=2)
        )
        self.persist = persist
        self.exclusion_radius_ratio = exclusion_radius_ratio
        self.confirm_frames = confirm_frames
        # Callback opcional que recibe el comando recien confirmado. Permite
        # que el visor reaccione (p. ej. mover el robot a una pose fija).
        self.on_command = on_command
        self.active = True
        self.last_command = "continue"
        self.last_gesture = "none"
        self._pending_command: str | None = None
        self._pending_count = 0
        # Instantanea para dibujar (la escribe update(), la lee draw()). Se
        # asigna de golpe para que sea segura entre hilos sin candado.
        self._draw_snapshot: tuple[
            GestureResult, tuple[float, float] | None, list[tuple[float, float]], float
        ] = (GestureResult("none", "none", 0.0), None, [], 0.0)

    def update(
        self,
        frame: Any,
        tool_pixels: list[tuple[float, float]] | None = None,
    ) -> GestureResult:
        """Analiza el frame y actualiza el estado (sin dibujar nada).

        Pensado para ejecutarse en un hilo aparte: no toca el frame que el
        hilo principal muestra. El dibujado se hace luego con :meth:`draw`.
        """
        tool_pixels = tool_pixels or []
        height, width = frame.shape[:2]
        radius = self.exclusion_radius_ratio * width

        chosen, centroid = self._select_control_hand(frame, tool_pixels, radius)
        if chosen is not None:
            result = classify_gesture(chosen)
            result.command = gesture_to_command(result.gesture, self.command_mapping)
            self._update_hysteresis(result)
        else:
            result = GestureResult("none", "none", 0.0)

        self._draw_snapshot = (result, centroid, tool_pixels, radius)
        return result

    def draw(self, frame: Any) -> None:
        """Dibuja el ultimo estado analizado sobre ``frame`` (overlay)."""
        result, centroid, tool_pixels, radius = self._draw_snapshot
        self._draw_state(frame, result, centroid, tool_pixels, radius)

    def process_frame(
        self,
        frame: Any,
        tool_pixels: list[tuple[float, float]] | None = None,
    ) -> GestureResult:
        """Analiza y dibuja sobre el mismo frame (modo secuencial, un hilo)."""
        result = self.update(frame, tool_pixels)
        self.draw(frame)
        return result

    def _select_control_hand(
        self,
        frame: Any,
        tool_pixels: list[tuple[float, float]],
        radius: float,
    ) -> tuple[Any | None, tuple[float, float] | None]:
        height, width = frame.shape[:2]
        hands = self.detector.detect(frame)
        mejor = None
        mejor_centroide: tuple[float, float] | None = None
        mejor_dist = -1.0
        for hand in hands:
            centroid = _hand_centroid_px(hand.points, width, height)
            if tool_pixels:
                dist_tool = min(
                    math.hypot(centroid[0] - tx, centroid[1] - ty)
                    for tx, ty in tool_pixels
                )
                if dist_tool <= radius:
                    # Mano pegada a la herramienta: es la que la sostiene.
                    continue
            else:
                dist_tool = float("inf")
            # Nos quedamos con la mano mas alejada de la herramienta.
            if dist_tool > mejor_dist:
                mejor_dist = dist_tool
                mejor = hand
                mejor_centroide = centroid
        return mejor, mejor_centroide

    def _update_hysteresis(self, result: GestureResult) -> None:
        if result.gesture == "unknown" or result.command == "none":
            self._pending_command = None
            self._pending_count = 0
            return
        if result.command == self._pending_command:
            self._pending_count += 1
        else:
            self._pending_command = result.command
            self._pending_count = 1
        if self._pending_count >= self.confirm_frames:
            self._apply(result)

    def _apply(self, result: GestureResult) -> None:
        if result.command == self.last_command:
            return
        # Solo 'continue' mantiene el seguimiento de la herramienta; cualquier
        # otro comando (stop, pause, ir a una pose...) lo detiene para que el
        # robot pueda quedarse quieto o ir a una posicion fija.
        self.active = result.command in _RESUME_COMMANDS
        self.last_command = result.command
        self.last_gesture = result.gesture
        if self.persist:
            from app_state import append_gesture

            append_gesture(
                {
                    "gesture": result.gesture,
                    "command": result.command,
                    "confidence": result.confidence,
                    "tracking_active": self.active,
                }
            )
        if self.on_command is not None:
            self.on_command(result.command)

    def _draw_state(
        self,
        frame: Any,
        result: GestureResult,
        centroid: tuple[float, float] | None,
        tool_pixels: list[tuple[float, float]],
        radius: float,
    ) -> None:
        if cv2 is None:
            return

        # Zona muerta alrededor de la herramienta (manos aqui se ignoran).
        for tx, ty in tool_pixels:
            cv2.circle(
                frame, (int(tx), int(ty)), int(radius), (80, 80, 80), 1, cv2.LINE_AA
            )

        # Mano de control elegida.
        if centroid is not None:
            cv2.circle(
                frame,
                (int(centroid[0]), int(centroid[1])),
                10,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        estado = "SIGUIENDO" if self.active else "PAUSADO"
        color = (0, 200, 0) if self.active else (0, 165, 255)
        # Progreso de confirmacion del gesto pendiente.
        progreso = ""
        if self._pending_command and self._pending_count < self.confirm_frames:
            progreso = f" (confirmando {self._pending_count}/{self.confirm_frames})"
        texto = f"Robot: {estado} | mano libre: {result.gesture} -> {self.last_command}{progreso}"
        cv2.putText(
            frame,
            texto,
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    def close(self) -> None:
        self.detector.close()


def _open_left_camera() -> Any:
    from vision.camera import CameraSource, load_stereo_camera_configs

    left_config, _ = load_stereo_camera_configs()
    camera = CameraSource(left_config)
    camera.open()
    return camera


def preview_gesture_detection() -> None:
    """Muestra camara + gesto clasificado para depuracion (no persiste)."""
    if cv2 is None:
        print("OpenCV no esta instalado. Instala opencv-contrib-python.")
        return

    from app_state import load_config

    config = load_config()
    mapping = config.get("gestures", {}).get("command_mapping", {})

    camera = _open_left_camera()
    detector = HandDetector()
    print("Previsualizacion de gestos. Pulsa 'q' para salir.")
    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue
            hands = detector.detect(frame)
            if hands:
                result = classify_gesture(hands[0])
                result.command = gesture_to_command(result.gesture, mapping)
            else:
                result = GestureResult("none", "none", 0.0)
            _draw_overlay(frame, result)
            cv2.imshow("Gestos (preview)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.close()
        camera.release()
        cv2.destroyAllWindows()


def run_gesture_detection() -> None:
    """Ejecuta deteccion de gestos y guarda el ultimo comando en `state/`."""
    if cv2 is None:
        print("OpenCV no esta instalado. Instala opencv-contrib-python.")
        return

    from app_state import append_gesture, load_config

    config = load_config()
    mapping = config.get("gestures", {}).get("command_mapping", {})

    camera = _open_left_camera()
    detector = HandDetector()
    ultimo_comando: str | None = None
    print("Deteccion de gestos. Pulsa 'q' para salir.")
    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue
            hands = detector.detect(frame)
            if hands:
                result = classify_gesture(hands[0])
                result.command = gesture_to_command(result.gesture, mapping)
            else:
                result = GestureResult("none", "none", 0.0)
            _draw_overlay(frame, result)
            cv2.imshow("Gestos", frame)

            # Solo persiste cuando el comando cambia, para no inundar el log.
            if result.command not in ("none", ultimo_comando):
                append_gesture(
                    {
                        "gesture": result.gesture,
                        "command": result.command,
                        "confidence": result.confidence,
                    }
                )
                ultimo_comando = result.command

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.close()
        camera.release()
        cv2.destroyAllWindows()

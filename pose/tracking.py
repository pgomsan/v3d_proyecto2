from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any


Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
Point3 = tuple[float, float, float]


@dataclass
class TemporalPoseResult:
    accepted: bool
    payload: dict[str, Any] | None
    reason: str


class TemporalPoseFilter:
    """Rechaza picos aislados y suaviza posicion/orientacion aceptadas."""

    def __init__(
        self,
        smoothing_alpha: float = 0.15,
        max_position_jump_cm: float = 5.0,
        max_orientation_jump_deg: float = 35.0,
        reacquire_frames: int = 3,
    ) -> None:
        self.smoothing_alpha = max(0.0, min(1.0, float(smoothing_alpha)))
        self.max_position_jump_cm = max(0.0, float(max_position_jump_cm))
        self.max_orientation_jump_deg = max(
            0.0, float(max_orientation_jump_deg)
        )
        self.reacquire_frames = max(1, int(reacquire_frames))
        self._last_observed_position: Point3 | None = None
        self._last_observed_rotation: Matrix3 | None = None
        self._filtered_position: Point3 | None = None
        self._filtered_rotation: Matrix3 | None = None
        self._consecutive_jumps = 0

    def filter(self, payload: dict[str, Any]) -> TemporalPoseResult:
        position = _payload_position(payload)
        rotation = _payload_rotation(payload)
        if position is None or rotation is None:
            return TemporalPoseResult(False, None, "Pose temporal incompleta")

        jump_note = "Pose estable"
        is_jump = False
        if (
            self._last_observed_position is not None
            and self._last_observed_rotation is not None
        ):
            position_jump = _distance_3d(
                self._last_observed_position, position
            )
            orientation_jump = _rotation_distance_deg(
                self._last_observed_rotation, rotation
            )
            if (
                position_jump > self.max_position_jump_cm
                or orientation_jump > self.max_orientation_jump_deg
            ):
                is_jump = True
                jump_note = (
                    f"Salto pos={position_jump:.1f}cm "
                    f"rot={orientation_jump:.1f}deg"
                )

        if is_jump:
            self._consecutive_jumps += 1
            if self._consecutive_jumps < self.reacquire_frames:
                # Rechazar pico aislado sin tocar el estado filtrado, asi
                # el visor mantiene la ultima pose buena.
                return TemporalPoseResult(
                    False, None, jump_note + " (descartado)"
                )
            # Salto sostenido durante reacquire_frames consecutivos: el
            # cambio es real (inversion A/B sostenida, reset rapido, etc.),
            # aceptarlo limpiando el filtro para que no haya inercia.
            jump_note = jump_note + " (reset por salto sostenido)"
            self._filtered_position = None
            self._filtered_rotation = None
            self._consecutive_jumps = 0
        else:
            self._consecutive_jumps = 0

        self._last_observed_position = position
        self._last_observed_rotation = rotation
        self._filtered_position = _smooth_point(
            self._filtered_position,
            position,
            self.smoothing_alpha,
        )
        self._filtered_rotation = _smooth_rotation(
            self._filtered_rotation,
            rotation,
            self.smoothing_alpha,
        )

        filtered_payload = copy.deepcopy(payload)
        filtered_payload["position_cm"] = list(self._filtered_position)
        filtered_payload["tip_position_cm"] = list(self._filtered_position)
        filtered_payload["direction"] = [
            self._filtered_rotation[0][0],
            self._filtered_rotation[1][0],
            self._filtered_rotation[2][0],
        ]
        filtered_payload["orientation"]["value"] = [
            list(row) for row in self._filtered_rotation
        ]
        filtered_payload["temporal_filter"] = {
            "smoothing_alpha": self.smoothing_alpha,
            "max_position_jump_cm": self.max_position_jump_cm,
            "max_orientation_jump_deg": self.max_orientation_jump_deg,
        }
        return TemporalPoseResult(True, filtered_payload, jump_note)


def _payload_position(payload: dict[str, Any]) -> Point3 | None:
    value = payload.get("position_cm")
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError, IndexError):
        return None


def _payload_rotation(payload: dict[str, Any]) -> Matrix3 | None:
    orientation = payload.get("orientation")
    if not isinstance(orientation, dict):
        return None
    if orientation.get("format") != "rotation_matrix":
        return None
    value = orientation.get("value")
    try:
        rows = tuple(
            tuple(float(value[row][column]) for column in range(3))
            for row in range(3)
        )
    except (TypeError, ValueError, IndexError):
        return None
    return rows  # type: ignore[return-value]


def _distance_3d(first: Point3, second: Point3) -> float:
    return math.sqrt(
        sum((first[index] - second[index]) ** 2 for index in range(3))
    )


def _rotation_distance_deg(first: Matrix3, second: Matrix3) -> float:
    relative_trace = sum(
        first[row][column] * second[row][column]
        for row in range(3)
        for column in range(3)
    )
    cosine = max(-1.0, min(1.0, (relative_trace - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def _smooth_point(
    previous: Point3 | None,
    current: Point3,
    alpha: float,
) -> Point3:
    if previous is None:
        return current
    return tuple(
        (1.0 - alpha) * previous[index] + alpha * current[index]
        for index in range(3)
    )  # type: ignore[return-value]


def _smooth_rotation(
    previous: Matrix3 | None,
    current: Matrix3,
    alpha: float,
) -> Matrix3:
    if previous is None:
        return current

    mixed = [
        [
            (1.0 - alpha) * previous[row][column]
            + alpha * current[row][column]
            for column in range(3)
        ]
        for row in range(3)
    ]
    x_axis = _normalize((mixed[0][0], mixed[1][0], mixed[2][0]))
    y_raw = (mixed[0][1], mixed[1][1], mixed[2][1])
    projection = _dot(x_axis, y_raw)
    y_axis = _normalize(
        tuple(y_raw[index] - projection * x_axis[index] for index in range(3))
    )
    z_axis = _normalize(_cross(x_axis, y_axis))
    y_axis = _cross(z_axis, x_axis)
    return (
        (x_axis[0], y_axis[0], z_axis[0]),
        (x_axis[1], y_axis[1], z_axis[1]),
        (x_axis[2], y_axis[2], z_axis[2]),
    )


def _dot(first: Point3, second: Point3) -> float:
    return sum(first[index] * second[index] for index in range(3))


def _cross(first: Point3, second: Point3) -> Point3:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _normalize(vector: Point3) -> Point3:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm <= 1e-12:
        raise ValueError("No se puede normalizar una orientacion nula.")
    return tuple(component / norm for component in vector)  # type: ignore[return-value]

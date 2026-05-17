from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

from pose.geometry import (
    Point3,
    build_orientation_from_direction,
    direction_from_points,
    distance_3d,
    midpoint_3d,
)


@dataclass
class ToolParameters:
    tool_id: str
    tool_type: str
    length_cm: float | None
    marker_distance_cm: float | None
    tip_offset_cm: tuple[float, float, float]


@dataclass
class ToolMarkerPair3D:
    marker_a_cm: Point3
    marker_b_cm: Point3
    confidence: float
    marker_pixels: dict[str, Any] | None = None


@dataclass
class ToolPose:
    tool_id: str
    tool_type: str
    frame: str
    position_cm: tuple[float, float, float]
    direction: tuple[float, float, float]
    orientation: Any
    confidence: float
    marker_center_cm: tuple[float, float, float] | None = None
    position_reference: str = "tool_tip"


class RodPoseEstimator:
    """Estimador de pose de la herramienta marcada.

    Este objeto deberia recibir detecciones de marcas ya limpias. No deberia
    abrir camaras ni detectar colores; su responsabilidad es solo geometrica.
    """

    def __init__(self, tool_parameters: ToolParameters) -> None:
        """Guarda parametros reales de la herramienta.

        Ejemplos: distancia real entre marcas, offset hasta la punta util,
        identificador de herramienta y tipo.
        """
        self.tool_parameters = tool_parameters

    def estimate_from_markers(self, marker_pair: Any) -> ToolPose | None:
        """Estima pose desde un par de marcas.

        Implementacion recomendada:
        - convertir centros de pixel a coordenadas 3D con el metodo elegido;
        - calcular posicion de referencia: centro, marca A o punta;
        - calcular direccion A -> B;
        - construir orientacion;
        - devolver `ToolPose` con confianza.
        """
        marker_a_cm, marker_b_cm = _marker_points_cm(marker_pair)
        if marker_a_cm is None or marker_b_cm is None:
            return None

        direction = direction_from_points(marker_a_cm, marker_b_cm)
        marker_center_cm = midpoint_3d(marker_a_cm, marker_b_cm)
        position_cm = _apply_tip_offset(
            marker_center_cm, direction, self.tool_parameters.tip_offset_cm
        )
        confidence = float(getattr(marker_pair, "confidence", 1.0))

        return ToolPose(
            tool_id=self.tool_parameters.tool_id,
            tool_type=self.tool_parameters.tool_type,
            frame="left_camera_rectified",
            position_cm=position_cm,
            direction=direction,
            orientation=build_orientation_from_direction(direction),
            confidence=confidence,
            marker_center_cm=marker_center_cm,
            position_reference="tool_tip",
        )

    def compute_tip_position(self, pose: ToolPose) -> tuple[float, float, float]:
        """Aplica `tip_offset_cm` para obtener la punta util de la herramienta."""
        if pose.marker_center_cm is None:
            return pose.position_cm
        return _apply_tip_offset(
            pose.marker_center_cm, pose.direction, self.tool_parameters.tip_offset_cm
        )

    def build_payload(self, pose: ToolPose, marker_pair: Any) -> dict[str, Any]:
        """Convierte la pose a JSON serializable para logs o RoboDK."""
        marker_a_cm, marker_b_cm = _marker_points_cm(marker_pair)
        marker_distance_cm = None
        if marker_a_cm is not None and marker_b_cm is not None:
            marker_distance_cm = distance_3d(marker_a_cm, marker_b_cm)

        payload: dict[str, Any] = {
            "frame": pose.frame,
            "units": "cm",
            "tool_id": pose.tool_id,
            "tool_type": pose.tool_type,
            "tool_parameters": _serializable_tool_parameters(self.tool_parameters),
            "position_reference": pose.position_reference,
            "position_cm": list(pose.position_cm),
            "tip_position_cm": list(self.compute_tip_position(pose)),
            "direction": list(pose.direction),
            "orientation": pose.orientation,
            "marker_distance_cm": marker_distance_cm,
            "expected_marker_distance_cm": self.tool_parameters.marker_distance_cm,
            "confidence": pose.confidence,
        }

        marker_pixels = getattr(marker_pair, "marker_pixels", None)
        markers: dict[str, Any] = {}
        if pose.marker_center_cm is not None:
            markers["center_3d_cm"] = list(pose.marker_center_cm)
        if marker_a_cm is not None:
            markers["a_3d_cm"] = list(marker_a_cm)
        if marker_b_cm is not None:
            markers["b_3d_cm"] = list(marker_b_cm)
        if marker_pixels:
            markers.update(_serializable_value(marker_pixels))
        payload["markers"] = markers
        return payload


def estimate_tool_pose(marker_pair: Any, tool_parameters: ToolParameters) -> ToolPose | None:
    """Atajo funcional para crear el estimador y calcular una pose."""
    return RodPoseEstimator(tool_parameters).estimate_from_markers(marker_pair)


def _marker_points_cm(marker_pair: Any) -> tuple[Point3 | None, Point3 | None]:
    if marker_pair is None:
        return None, None

    if isinstance(marker_pair, dict):
        marker_a = marker_pair.get("marker_a_cm") or marker_pair.get("a_3d_cm")
        marker_b = marker_pair.get("marker_b_cm") or marker_pair.get("b_3d_cm")
    else:
        marker_a = getattr(marker_pair, "marker_a_cm", None) or getattr(
            marker_pair, "a_3d_cm", None
        )
        marker_b = getattr(marker_pair, "marker_b_cm", None) or getattr(
            marker_pair, "b_3d_cm", None
        )

    if marker_a is None or marker_b is None:
        return None, None
    return _as_point3(marker_a), _as_point3(marker_b)


def _as_point3(point: Any) -> Point3:
    return (float(point[0]), float(point[1]), float(point[2]))


def _apply_tip_offset(
    marker_center_cm: Point3,
    direction: tuple[float, float, float],
    tip_offset_cm: tuple[float, float, float],
) -> Point3:
    offset_along_tool, offset_y, offset_z = tip_offset_cm
    return (
        marker_center_cm[0] + direction[0] * offset_along_tool,
        marker_center_cm[1] + direction[1] * offset_along_tool + offset_y,
        marker_center_cm[2] + direction[2] * offset_along_tool + offset_z,
    )


def _serializable_tool_parameters(tool_parameters: ToolParameters) -> dict[str, Any]:
    payload = asdict(tool_parameters)
    payload["tip_offset_cm"] = list(tool_parameters.tip_offset_cm)
    return payload


def _serializable_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _serializable_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_serializable_value(item) for item in value]
    if isinstance(value, list):
        return [_serializable_value(item) for item in value]
    return value

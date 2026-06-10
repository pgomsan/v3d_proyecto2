from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

from pose.geometry import (
    Point3,
    build_orientation_from_markers,
    distance_3d,
    midpoint_3d,
    rotation_matrix_from_markers,
)


@dataclass
class ToolParameters:
    tool_id: str
    tool_type: str
    length_cm: float | None
    marker_distance_cm: float | None
    tip_offset_cm: tuple[float, float, float]
    marker_c_along_ab_cm: float | None = None
    marker_c_offset_cm: float | None = None
    marker_distance_tolerance_ratio: float = 0.50


@dataclass
class ToolMarkerTriplet3D:
    marker_a_cm: Point3
    marker_b_cm: Point3
    marker_c_cm: Point3
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

    def estimate_from_markers(self, marker_triplet: Any) -> ToolPose | None:
        """Estima posicion y orientacion completa desde A, B y C."""
        marker_a_cm, marker_b_cm, marker_c_cm = _marker_points_cm(marker_triplet)
        if marker_a_cm is None or marker_b_cm is None or marker_c_cm is None:
            return None

        measured_distances = _marker_distances(
            marker_a_cm, marker_b_cm, marker_c_cm
        )
        expected_distances = _expected_marker_distances(self.tool_parameters)
        if not _distances_are_valid(
            measured_distances,
            expected_distances,
            self.tool_parameters.marker_distance_tolerance_ratio,
        ):
            return None

        try:
            rotation_matrix = rotation_matrix_from_markers(
                marker_a_cm, marker_b_cm, marker_c_cm
            )
        except ValueError:
            return None

        direction = (
            rotation_matrix[0][0],
            rotation_matrix[1][0],
            rotation_matrix[2][0],
        )
        marker_center_cm = midpoint_3d(marker_a_cm, marker_b_cm)
        position_cm = _apply_local_offset(
            marker_a_cm,
            rotation_matrix,
            self.tool_parameters.tip_offset_cm,
        )
        confidence = float(getattr(marker_triplet, "confidence", 1.0))

        return ToolPose(
            tool_id=self.tool_parameters.tool_id,
            tool_type=self.tool_parameters.tool_type,
            frame="left_camera_rectified",
            position_cm=position_cm,
            direction=direction,
            orientation=build_orientation_from_markers(
                marker_a_cm, marker_b_cm, marker_c_cm
            ),
            confidence=confidence,
            marker_center_cm=marker_center_cm,
            position_reference="marker_a_tool_tip",
        )

    def compute_tip_position(self, pose: ToolPose) -> tuple[float, float, float]:
        """Devuelve el TCP, calculado desde A al estimar la pose."""
        return pose.position_cm

    def build_payload(self, pose: ToolPose, marker_triplet: Any) -> dict[str, Any]:
        """Convierte la pose a JSON serializable para logs o RoboDK."""
        marker_a_cm, marker_b_cm, marker_c_cm = _marker_points_cm(marker_triplet)
        marker_distances: dict[str, float] = {}
        if marker_a_cm is not None and marker_b_cm is not None and marker_c_cm is not None:
            marker_distances = _marker_distances(
                marker_a_cm, marker_b_cm, marker_c_cm
            )
        expected_distances = _expected_marker_distances(self.tool_parameters)

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
            "marker_distance_cm": marker_distances.get("ab"),
            "expected_marker_distance_cm": self.tool_parameters.marker_distance_cm,
            "marker_distances_cm": marker_distances,
            "expected_marker_distances_cm": expected_distances,
            "confidence": pose.confidence,
        }

        marker_pixels = getattr(marker_triplet, "marker_pixels", None)
        markers: dict[str, Any] = {}
        if pose.marker_center_cm is not None:
            markers["center_3d_cm"] = list(pose.marker_center_cm)
        if marker_a_cm is not None:
            markers["a_3d_cm"] = list(marker_a_cm)
        if marker_b_cm is not None:
            markers["b_3d_cm"] = list(marker_b_cm)
        if marker_c_cm is not None:
            markers["c_3d_cm"] = list(marker_c_cm)
        if marker_pixels:
            markers.update(_serializable_value(marker_pixels))
        payload["markers"] = markers
        return payload


def estimate_tool_pose(marker_triplet: Any, tool_parameters: ToolParameters) -> ToolPose | None:
    """Atajo funcional para crear el estimador y calcular una pose."""
    return RodPoseEstimator(tool_parameters).estimate_from_markers(marker_triplet)


def _marker_points_cm(
    marker_triplet: Any,
) -> tuple[Point3 | None, Point3 | None, Point3 | None]:
    if marker_triplet is None:
        return None, None, None

    if isinstance(marker_triplet, dict):
        marker_a = marker_triplet.get("marker_a_cm") or marker_triplet.get("a_3d_cm")
        marker_b = marker_triplet.get("marker_b_cm") or marker_triplet.get("b_3d_cm")
        marker_c = marker_triplet.get("marker_c_cm") or marker_triplet.get("c_3d_cm")
    else:
        marker_a = getattr(marker_triplet, "marker_a_cm", None) or getattr(
            marker_triplet, "a_3d_cm", None
        )
        marker_b = getattr(marker_triplet, "marker_b_cm", None) or getattr(
            marker_triplet, "b_3d_cm", None
        )
        marker_c = getattr(marker_triplet, "marker_c_cm", None) or getattr(
            marker_triplet, "c_3d_cm", None
        )

    if marker_a is None or marker_b is None or marker_c is None:
        return None, None, None
    return _as_point3(marker_a), _as_point3(marker_b), _as_point3(marker_c)


def _as_point3(point: Any) -> Point3:
    return (float(point[0]), float(point[1]), float(point[2]))


def _marker_distances(
    marker_a_cm: Point3,
    marker_b_cm: Point3,
    marker_c_cm: Point3,
) -> dict[str, float]:
    return {
        "ab": distance_3d(marker_a_cm, marker_b_cm),
        "ac": distance_3d(marker_a_cm, marker_c_cm),
        "bc": distance_3d(marker_b_cm, marker_c_cm),
    }


def _expected_marker_distances(
    tool_parameters: ToolParameters,
) -> dict[str, float]:
    ab = tool_parameters.marker_distance_cm
    along = tool_parameters.marker_c_along_ab_cm
    offset = tool_parameters.marker_c_offset_cm
    if ab is None or along is None or offset is None:
        return {}
    return {
        "ab": float(ab),
        "ac": (float(along) ** 2 + float(offset) ** 2) ** 0.5,
        "bc": ((float(ab) - float(along)) ** 2 + float(offset) ** 2) ** 0.5,
    }


def _distances_are_valid(
    measured: dict[str, float],
    expected: dict[str, float],
    tolerance_ratio: float,
) -> bool:
    for key, expected_distance in expected.items():
        if expected_distance <= 0.0:
            continue
        tolerance = max(float(tolerance_ratio) * expected_distance, 1.0)
        if abs(measured[key] - expected_distance) > tolerance:
            return False
    return True


def _apply_local_offset(
    origin_cm: Point3,
    rotation_matrix: tuple[Point3, Point3, Point3],
    tip_offset_cm: tuple[float, float, float],
) -> Point3:
    ox, oy, oz = tip_offset_cm
    return (
        origin_cm[0]
        + rotation_matrix[0][0] * ox
        + rotation_matrix[0][1] * oy
        + rotation_matrix[0][2] * oz,
        origin_cm[1]
        + rotation_matrix[1][0] * ox
        + rotation_matrix[1][1] * oy
        + rotation_matrix[1][2] * oz,
        origin_cm[2]
        + rotation_matrix[2][0] * ox
        + rotation_matrix[2][1] * oy
        + rotation_matrix[2][2] * oz,
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

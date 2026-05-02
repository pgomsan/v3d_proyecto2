from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolParameters:
    tool_id: str
    tool_type: str
    length_cm: float | None
    marker_distance_cm: float | None
    tip_offset_cm: tuple[float, float, float]


@dataclass
class ToolPose:
    tool_id: str
    frame: str
    position_cm: tuple[float, float, float]
    direction: tuple[float, float, float]
    orientation: Any
    confidence: float


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
        pass

    def estimate_from_markers(self, marker_pair: Any) -> ToolPose | None:
        """Estima pose desde un par de marcas.

        Implementacion recomendada:
        - convertir centros de pixel a coordenadas 3D con el metodo elegido;
        - calcular posicion de referencia: centro, marca A o punta;
        - calcular direccion A -> B;
        - construir orientacion;
        - devolver `ToolPose` con confianza.
        """
        pass

    def compute_tip_position(self, pose: ToolPose) -> tuple[float, float, float]:
        """Aplica `tip_offset_cm` para obtener la punta util de la herramienta."""
        pass

    def build_payload(self, pose: ToolPose, marker_pair: Any) -> dict[str, Any]:
        """Convierte la pose a JSON serializable para logs o RoboDK."""
        pass


def estimate_tool_pose(marker_pair: Any, tool_parameters: ToolParameters) -> ToolPose | None:
    """Atajo funcional para crear el estimador y calcular una pose."""
    pass

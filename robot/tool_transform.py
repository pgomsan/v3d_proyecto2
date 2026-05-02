from __future__ import annotations

from typing import Any


def build_robodk_pose_payload(tool_pose: Any, gesture_command: Any | None = None) -> dict[str, Any]:
    """Prepara el payload que consumira el proyecto de RoboDK.

    Esta funcion no debe llamar a RoboDK. Solo empaqueta datos:
    - pose de herramienta;
    - tipo e identificador;
    - comando de gesto activo;
    - frame/unidades/formato de orientacion.
    """
    pass


def export_robodk_payload(payload: dict[str, Any]) -> None:
    """Exporta el payload por el mecanismo acordado con el compañero.

    Opciones posibles:
    - escribir `state/robodk_payload.json`;
    - publicar por MQTT;
    - socket local;
    - API HTTP local.
    """
    pass

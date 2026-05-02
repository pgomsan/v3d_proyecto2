from __future__ import annotations


def preview_marker_detection() -> None:
    """Previsualiza deteccion de marcas sin estimar pose.

    Deberia abrir las dos camaras, ejecutar solo `vision/color_markers.py` y
    dibujar resultados de depuracion. Util para ajustar HSV.
    """
    pass


def run_pose_estimation() -> None:
    """Bucle principal de estimacion de pose.

    Implementacion recomendada:
    - abrir camaras izquierda y derecha;
    - leer frames sincronizados de forma aproximada;
    - corregir/rectificar si ya tienes calibracion;
    - detectar marcas;
    - reconstruir posicion 3D con tu metodo;
    - guardar pose actual y log.
    """
    pass


def run_pose_and_gestures() -> None:
    """Ejecuta pose y gestos en el mismo bucle.

    Esta funcion deberia producir un payload conjunto para RoboDK:
    pose de herramienta + comando de gesto activo.
    """
    pass

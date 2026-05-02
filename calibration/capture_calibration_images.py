from __future__ import annotations


def capture_calibration_images() -> None:
    """Captura pares de imagenes para calibracion estereo.

    Implementacion recomendada:
    - cargar `cameras.left` y `cameras.right` desde `state/config.json`;
    - abrir ambas camaras con `CameraSource`;
    - mostrar los dos frames en ventanas OpenCV;
    - al pulsar una tecla, guardar un par sincronizado:
      `data/calibration_images/left_XX.png` y `right_XX.png`;
    - guardar siempre pares con el mismo indice para poder calibrar estereo.
    """
    pass

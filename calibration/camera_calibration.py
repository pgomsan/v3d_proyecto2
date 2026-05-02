from __future__ import annotations


def calibrate_camera() -> None:
    """Calcula parametros intrinsecos para las dos camaras.

    Implementacion recomendada:
    - leer las imagenes `left_XX.png` y `right_XX.png`;
    - detectar esquinas del tablero en cada camara;
    - llamar a `cv2.calibrateCamera` por separado para izquierda y derecha;
    - guardar `K_left.npy`, `dist_left.npy`, `K_right.npy`, `dist_right.npy`;
    - si haces calibracion estereo, usar `cv2.stereoCalibrate` y guardar
      rotacion/traslacion entre camaras en `calibration/stereo.npz`;
    - actualizar `state/calibration_info.json` con error de reproyeccion.
    """
    pass

from __future__ import annotations


def compute_homography() -> None:
    """Calcula la homografia del plano de trabajo si decides usar plano conocido.

    Implementacion recomendada:
    - cargar los puntos seleccionados en imagen;
    - cargar o pedir las coordenadas reales del plano;
    - usar `cv2.findHomography` o `cv2.getPerspectiveTransform`;
    - guardar `calibration/H.npy`;
    - guardar un resumen en `state/calibration_info.json`.

    Si finalmente haces reconstruccion estereo pura, esta funcion puede quedar
    como utilidad secundaria para depuracion planar.
    """
    pass

from __future__ import annotations


def select_points() -> None:
    """Selecciona puntos manuales necesarios para tu modelo geometrico.

    Implementacion recomendada:
    - abrir un frame de la camara que uses como referencia;
    - permitir click de puntos con `cv2.setMouseCallback`;
    - guardar puntos de imagen y puntos reales asociados en `state/`;
    - si luego vas a hacer homografia planar, mantener siempre el mismo orden:
      arriba-izquierda, arriba-derecha, abajo-derecha, abajo-izquierda.
    """
    pass

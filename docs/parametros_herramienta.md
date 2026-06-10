# Parametros de herramienta

El objetivo es que una persona maneje una herramienta fisica y que el sistema estime los parametros necesarios para que el robot pueda imitar ese movimiento en RoboDK.

## Ejemplo: bisturi

Parametros minimos a medir:

- `tool_id`: identificador de la herramienta, por ejemplo `bisturi_01`.
- `tool_type`: tipo de herramienta, por ejemplo `bisturi`.
- `length_cm`: longitud total aproximada.
- `marker_distance_cm`: distancia real A-B.
- `marker_c_along_ab_cm`: posicion de la rama C medida desde A sobre A-B.
- `marker_c_offset_cm`: longitud perpendicular de la rama hasta C.
- `marker_a_color`: color de la primera marca.
- `marker_b_color`: color de la segunda marca.
- `marker_c_color`: color de la tercera marca.
- `tip_offset_cm`: vector desde el punto de referencia estimado hasta la punta util de la herramienta.

## Medidas actuales

- Herramienta rigida con tres ramas y tres marcadores.
- A: rosa, punta util/TCP y origen local.
- B: verde, a 15 cm de A.
- C: amarillo, sobre una rama que nace a 5 cm de A y mide 5 cm.
- Modelo local: `A=(0,0,0)`, `B=(16.5,0,0)`, `C=(5.8,5.8,0)`.
- Distancias esperadas: `AB=15 cm`, `AC=7.07 cm`, `BC=11.18 cm`.
- `tip_offset_cm`: `[0,0,0]`, porque el TCP coincide con el centro de A.
- Eje X local: A -> B.
- Eje Y local: hacia C, ortogonalizado respecto a X.
- Eje Z local: X x Y.
- Rango HSV inicial para rosa: `[145, 60, 80] -> [179, 255, 255]`.
- Rango HSV inicial para verde: `[40, 100, 60] -> [88, 255, 255]`.
- Rango HSV inicial para amarillo: `[18, 100, 100] -> [40, 255, 255]`.

## Punto de referencia

`position_cm` y `tip_position_cm` representan el centro de la marca A, que
coincide con la punta util/TCP. Las tres marcas permiten recuperar una
orientacion completa 3D en lugar de solo el vector principal.

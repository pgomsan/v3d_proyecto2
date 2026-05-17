# Parametros de herramienta

El objetivo es que una persona maneje una herramienta fisica y que el sistema estime los parametros necesarios para que el robot pueda imitar ese movimiento en RoboDK.

## Ejemplo: bisturi

Parametros minimos a medir:

- `tool_id`: identificador de la herramienta, por ejemplo `bisturi_01`.
- `tool_type`: tipo de herramienta, por ejemplo `bisturi`.
- `length_cm`: longitud total aproximada.
- `marker_distance_cm`: distancia real entre las dos marcas de color.
- `marker_a_color`: color de la primera marca.
- `marker_b_color`: color de la segunda marca.
- `tip_offset_cm`: vector desde el punto de referencia estimado hasta la punta util de la herramienta.

## Medidas actuales

- Herramienta usada: boligrafo con marca verde y marca rosa.
- `marker_distance_cm`: 8.0 cm entre la marca verde y la rosa.
- `marker_a_color`: verde.
- `marker_b_color`: rosa.
- `tip_offset_cm`: `[-7.5, 0.0, 0.0]`, desde el centro entre marcas hasta la punta, en sentido contrario al vector verde -> rosa.
- Rango HSV inicial para verde en OpenCV: `[55, 100, 60] -> [88, 255, 255]`.
- Rango HSV inicial para rosa en OpenCV: `[145, 60, 80] -> [179, 255, 255]`.

## Punto de referencia

Antes de programar la pose conviene decidir que punto representa `position_cm`: punta de la herramienta.

- centro entre las dos marcas;
- marca A;
- marca B;
- punta util de la herramienta.

Para RoboDK lo mas util normalmente sera la punta de la herramienta, pero puede ser mas facil estimar primero el centro entre marcas y aplicar despues `tip_offset_cm`.

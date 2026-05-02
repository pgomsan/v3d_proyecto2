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

## Punto de referencia

Antes de programar la pose conviene decidir que punto representa `position_cm`:

- centro entre las dos marcas;
- marca A;
- marca B;
- punta util de la herramienta.

Para RoboDK lo mas util normalmente sera la punta de la herramienta, pero puede ser mas facil estimar primero el centro entre marcas y aplicar despues `tip_offset_cm`.

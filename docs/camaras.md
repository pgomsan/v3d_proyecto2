# Camaras

Datos anotados a partir de la caja de la camara Trust Trino.

## Modelo

- Marca: Trust
- Modelo: Trino
- Tipo: HD Webcam for Video Calls
- Microfono: integrado
- Boton de foto: integrado
- Montaje: soporte universal para portatil, monitor o escritorio
- Conexion: USB

## Especificaciones indicadas por fabricante

- Resolucion de video: 1280 x 720 px
- FPS indicado: 30 frames por segundo
- Calidad: HD 720p

## Configuracion inicial del proyecto

| Rol | Modelo | Indice OpenCV | Resolucion configurada | FPS configurado | FPS real medido |
| --- | --- | ---: | --- | ---: | --- |
| Izquierda | Trust Trino HD Webcam | 0 | 1280 x 720 | 30 | Pendiente de medir en OpenCV |
| Derecha | Trust Trino HD Webcam | 1 | 1280 x 720 | 30 | Pendiente de medir en OpenCV |

## Nota sobre FPS real

El valor de 30 FPS viene indicado en la caja. El FPS real depende de iluminacion, puerto USB, ordenador, backend de OpenCV y si se abren una o dos camaras a la vez.

Para considerar esta tarea completamente validada, hay que medir el FPS desde el programa con ambas camaras abiertas y actualizar la columna `FPS real medido`.

## Comprobaciones pendientes

- Confirmar que las dos camaras fisicas son el mismo modelo.
- Confirmar indices reales despues de ejecutar `Buscar y elegir camaras`.
- Medir FPS real con ambas camaras abiertas.
- Anotar si OpenCV acepta realmente 1280 x 720 o reduce la resolucion.

# Interfaz con RoboDK

La simulacion y la parte del robot las desarrollara otro compañero en RoboDK. Este proyecto solo debe entregar informacion de vision 3D.

## Responsabilidad de este proyecto

- Detectar las tres marcas de color.
- Estimar posicion y orientacion del palo/herramienta.
- Asociar la pose a una herramienta parametrizada, por ejemplo un bisturi.
- Detectar gestos de mano y traducirlos a comandos de alto nivel.
- Guardar la ultima pose y un historico.
- Documentar el formato de salida.

## Fuera de alcance

- Control del robot.
- Cinemática directa o inversa.
- Planificacion de trayectorias.
- Simulacion en RoboDK.
- Calibracion entre camara y robot, salvo guardar los parametros si el compañero los proporciona.

## Formato propuesto de pose

```json
{
  "timestamp": "2026-04-30T17:00:00",
  "frame": "camera",
  "tool_id": "bisturi_01",
  "tool_type": "bisturi",
  "tool_parameters": {
    "length_cm": null,
    "marker_distance_cm": 16.5,
    "marker_c_along_ab_cm": 5.8,
    "marker_c_offset_cm": 5.8,
    "tip_offset_cm": [0.0, 0.0, 0.0]
  },
  "position_cm": [0.0, 0.0, 0.0],
  "direction": [1.0, 0.0, 0.0],
  "orientation": {
    "format": "rotation_matrix",
    "value": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
  },
  "markers": {
    "a_3d_cm": [0, 0, 0],
    "b_3d_cm": [16.5, 0, 0],
    "c_3d_cm": [5.8, 5.8, 0]
  },
  "confidence": 0.0
}
```

## Formato propuesto de gesto

```json
{
  "timestamp": "2026-04-30T17:00:00",
  "gesture": "open_hand",
  "command": "stop",
  "confidence": 0.0
}
```

## Comandos previstos

- `stop`: el robot debe detenerse.
- `continue`: el robot puede continuar.
- `pause`: el robot debe pausar sin cancelar.
- `none`: no hay orden nueva.

## Punto de acuerdo con el compañero

Antes de integrar, conviene fijar:

- unidades: centimetros o milimetros;
- sistema de referencia: camara, mesa, herramienta o robot;
- formato de orientacion: actualmente matriz 3x3 con columnas X/Y/Z;
- punto seguido por el robot: centro de A, definido como punta/TCP;
- metodo de intercambio: archivo JSON, socket, MQTT, API local o lectura directa de log.

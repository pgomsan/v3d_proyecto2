# Proyecto 2 - Pose de herramienta con marcas de color y gestos

Base limpia para desarrollar desde cero una aplicacion de vision 3D.

## Objetivo

Estimar la posicion y orientacion completa de una herramienta usando tres
marcas de colores. A y B definen el eje principal y C elimina la ambiguedad
de giro alrededor de ese eje.

El caso de uso final es que una persona mueva una herramienta fisica, por ejemplo un bisturi, y el sistema obtenga los parametros necesarios para que el robot pueda imitar ese manejo en RoboDK.

El proyecto tambien reserva una parte independiente para deteccion de gestos de la mano. Algunos gestos se usaran como comandos de alto nivel, por ejemplo detener o continuar la ejecucion del robot. La parte de robot queda fuera de este proyecto: la hara otro compañero en una plataforma de simulacion como RoboDK.

## Estructura

- `app.py`: menu principal de la aplicacion.
- `app_state.py`: persistencia de configuracion, calibracion, ultima pose y logs.
- `main_pose.py`: futuro bucle principal de camaras, deteccion de marcas y estimacion de pose.
- `main_gestures.py`: futuro bucle principal de deteccion de gestos.
- `calibration/`: scripts de calibracion de camara y plano.
- `vision/`: captura de camara, deteccion por color y visualizacion de depuracion.
- `pose/`: geometria y calculo de posicion/orientacion del palo.
- `gestures/`: deteccion de manos y clasificacion de gestos.
- `robot/`: interfaz minima de salida hacia el trabajo del compañero; no contiene logica de robot.
- `state/`: archivos persistentes generados o editados por la aplicacion.
- `data/`: imagenes de calibracion, muestras y grabaciones.
- `tests/`: pruebas unitarias para funciones puras.
- `docs/`: notas de arquitectura, decisiones y experimentos.

## Flujo recomendado

1. Seleccionar camara izquierda y camara derecha desde el menu.
2. Confirmar con `Ver camaras` que ambas se abren correctamente.
3. Capturar pares de imagenes de calibracion.
4. Calibrar ambas camaras y guardar parametros intrinsecos.
5. Definir el modelo geometrico de la herramienta: A rosa como TCP, B verde a 15 cm y C amarilla en `(5,5,0)` cm.
6. Detectar las tres marcas por color en imagen.
7. Convertir las detecciones de imagen a posicion 3D con el modelo elegido.
8. Calcular una matriz de orientacion a partir del frame A-B-C.
9. Guardar ultima pose y log historico en `state/`.
10. Anadir deteccion de gestos como modulo independiente.
11. Convertir gestos concretos en comandos como `stop`, `continue` o `pause`.
12. Exportar pose, parametros de herramienta y comando activo en un formato claro para que el compañero lo use en RoboDK.

## Persistencia prevista

- `state/config.json`: parametros editables del proyecto.
- `state/calibration_info.json`: resumen de la calibracion usada.
- `state/last_pose.json`: ultima pose estimada del palo/herramienta.
- `state/last_gesture.json`: ultimo gesto detectado.
- `state/logs/poses.jsonl`: historico de poses.
- `state/logs/gestures.jsonl`: historico de gestos.

## Arranque basico

Desde `Proyecto2`:

```bash
python3 app.py
```

Opciones que ya tienen infraestructura basica:

- `Buscar y elegir camaras`: busca indices OpenCV y guarda izquierda/derecha.
- `Ver camaras`: abre una ventana para cada camara.
- `Configurar herramienta`: guarda parametros del bisturi/herramienta.
- `Configurar comandos de gestos`: guarda el mapa gesto -> comando.
- `Ver configuracion`: imprime el JSON actual.

## Interfaz con RoboDK

Este proyecto solo debe producir datos. El consumidor sera el proyecto de simulacion del compañero.

Salida recomendada de pose:

```json
{
  "timestamp": "2026-04-30T17:00:00",
  "frame": "camera",
  "tool_id": "bisturi_01",
  "tool_type": "bisturi",
  "position_cm": [0.0, 0.0, 0.0],
  "direction": [1.0, 0.0, 0.0],
  "orientation": {
    "format": "rotation_matrix",
    "value": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
  },
  "markers": {
    "a_3d_cm": [0, 0, 0],
    "b_3d_cm": [15, 0, 0],
    "c_3d_cm": [5, 5, 0]
  },
  "confidence": 0.0
}
```

Salida recomendada de gesto/comando:

```json
{
  "timestamp": "2026-04-30T17:00:00",
  "gesture": "open_hand",
  "command": "stop",
  "confidence": 0.0
}
```

La conversion final al sistema de referencia del robot o de RoboDK no se implementa aqui salvo que se acuerde una interfaz concreta.

## Dependencias

Crear entorno e instalar:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Nota tecnica

Con una sola camara, dos marcas 2D no bastan por si solas para recuperar una pose 3D completa sin asumir algo mas. Antes de programar la estimacion conviene decidir si usaras plano conocido, distancia/longitud conocida, camara de profundidad, dos camaras, ZED, o un modelo PnP con puntos 3D conocidos.

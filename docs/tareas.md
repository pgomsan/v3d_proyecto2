# Tareas

## Hecho

- Estructura limpia del proyecto.
- Interfaz principal tipo `Proyecto1`.
- Configuracion persistente en `state/config.json`.
- Seleccion de dos camaras: izquierda y derecha.
- Previsualizacion basica de las dos camaras.
- Esqueletos comentados para calibracion, vision, pose, gestos y salida RoboDK.
- Persistencia basica de configuracion, ultima pose, ultimo gesto y logs.

## Siguiente paso recomendado

- Probar desde el menu `Buscar y elegir camaras`.
- Abrir `Ver camaras` y confirmar que izquierda/derecha son correctas.
- Si estan cruzadas, intercambiar indices en el menu.

## Pendiente de implementar

- Captura sincronizada de pares de calibracion.
- Calibracion intrinseca de camara izquierda y derecha.
- Calibracion estereo entre ambas camaras.
- Homografia si finalmente usas plano de trabajo.
- Ajuste HSV de las dos marcas de color.
- Deteccion de marcas en `vision/color_markers.py`.
- Reconstruccion 3D o modelo geometrico elegido.
- Geometria pura en `pose/geometry.py`.
- Estimacion de pose en `pose/rod_pose.py`.
- Deteccion de manos en `gestures/hand_detector.py`.
- Clasificacion de gestos en `gestures/gesture_classifier.py`.
- Guardado real de poses desde el bucle principal.
- Exportacion del payload acordado con el compañero de RoboDK.

## Fuera de alcance

- Cinematica del robot.
- Trayectorias.
- Control del robot.
- Simulacion en RoboDK.

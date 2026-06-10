# Arquitectura inicial

## Modulos

- `vision`: entrada de camaras izquierda/derecha y deteccion de marcas.
- `pose`: calculo geometrico de posicion y orientacion.
- `gestures`: deteccion y clasificacion de gestos.
- `robot`: salida minima para integracion externa; la simulacion y logica del robot quedan fuera de este proyecto.
- `app_state`: lectura y escritura del estado persistente.

## Pipeline de pose

1. Solicitar ambos frames con `grab()` y recuperarlos despues con `retrieve()`
   para reducir el desfase temporal.
2. Corregir distorsion y rectificar si las camaras estan calibradas.
3. Segmentar marcas A, B y C por color en ambos frames.
4. Obtener el centro de cada marca en pixeles.
5. Rechazar correspondencias cuyo error vertical epipolar supere el limite operativo.
6. Convertir centros validos a coordenadas 3D con triangulacion.
7. Construir el frame local con X=`A -> B`, Y hacia C y Z=`X x Y`.
8. Construir pose de la herramienta con TCP en A y orientacion 3D completa.
9. Rechazar saltos temporales aislados y suavizar posicion/orientacion.
10. Guardar y enviar al visor solo poses con estado `VALID`.
11. Asociar la pose a un modelo de herramienta, por ejemplo `bisturi_01`.
12. Exportar la pose en un formato estable para RoboDK.

## Pipeline de gestos

1. Leer frame de la camara principal elegida.
2. Detectar mano y landmarks.
3. Clasificar gesto.
4. Traducir gestos concretos a comandos de alto nivel.
5. Guardar ultimo gesto/comando y log historico.

## Comandos por gestos

Los gestos no deben mover el robot directamente en este proyecto. Deben producir comandos simples para el proyecto de RoboDK:

- `stop`: detener ejecucion.
- `continue`: continuar ejecucion.
- `pause`: pausar temporalmente.
- `none`: sin comando activo.

## Limite con el trabajo de RoboDK

Este proyecto no debe implementar movimiento del robot, cinemática, trayectorias ni control. La responsabilidad aqui es producir una pose limpia de la herramienta y, si hace falta, documentar el formato para que el compañero la lea desde RoboDK.

## Decision pendiente

La reconstruccion 3D depende del modelo de captura. La infraestructura ya permite seleccionar dos camaras, pero aun hay que decidir el metodo exacto:

- plano de trabajo conocido;
- longitud real del palo y restricciones geometricas;
- camara RGB-D o ZED;
- dos camaras calibradas;
- PnP con varios puntos 3D conocidos.

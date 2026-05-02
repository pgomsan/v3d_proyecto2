# Arquitectura inicial

## Modulos

- `vision`: entrada de camaras izquierda/derecha y deteccion de marcas.
- `pose`: calculo geometrico de posicion y orientacion.
- `gestures`: deteccion y clasificacion de gestos.
- `robot`: salida minima para integracion externa; la simulacion y logica del robot quedan fuera de este proyecto.
- `app_state`: lectura y escritura del estado persistente.

## Pipeline de pose

1. Leer frame de camara izquierda y derecha.
2. Corregir distorsion y rectificar si las camaras estan calibradas.
3. Segmentar marca A y marca B por color en los frames necesarios.
4. Obtener centro de cada marca en pixeles.
5. Convertir centros a coordenadas 3D con el modelo elegido.
6. Calcular vector `A -> B`.
7. Construir pose de la herramienta: posicion, direccion y orientacion.
8. Guardar la ultima pose y anadir una linea al log.
9. Asociar la pose a un modelo de herramienta, por ejemplo `bisturi_01`.
10. Exportar la pose en un formato estable para RoboDK.

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

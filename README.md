Está hecho pensando en usuarios de nivel medio.

Descargar los archivos en una misma carpeta.

## Qué se necesita si soy usuario de Windows 10?

Se necesita tener instalado Python 3.9 (preferiblemente), el cual puede descargarse desde la tienda oficial de Microsoft de forma gratuita.

- [Python 3.9](https://www.microsoft.com/store/productId/9P7QFQMJRFP7)

- [Guía de instalación de ffmpeg y variables de entorno](https://www.youtube.com/watch?v=r1AtmY-RMyQ)

También es necesario instalar algunas dependencias de Python la primera vez para el correcto funcionamiento del script, para esto tan sólo se debe dar doble click al archivo: `instalar dependencias.bat`. Si por el contrario, se desean instalar las dependencias de Python de forma manual, es necesario ejecutar las siguientes líneas de código en el prompt (parece repetir, pero es una forma de evitar errores si hay más de una versión de python en el pc, o más de un usuario incluso en windows):

```sh
pip3 install pandas
python3 -m pip install --upgrade pip
pip3 install matplotlib
python -m pip install --upgrade pip
python3 -m pip install matplotlib --user
```

## Cómo ejecuto el script de Python?
 
Para ejecutar el script, basta con darle doble click al archivo: `Run main.bat`. Aunque si se desea ejecutar el script manualmente, hay que abrir el prompt de Windows, navegar hasta la dirección en que están contenidos los archivos, y ejecutar:

```sh
python3 main.py
```

## Explicación Conceptual del perceptrón y red neuronal
 
Este script de Python dibuja la configuración de la red neuronal y además entrena y calcula una nueva situación. Tan sólo hace que configurar el archivo **configurationNetwork** donde se colocar la data de entrenamiento y la cantidad de neuronas y capas de la red. Ojo, la cantidad de neuronas en la capas input y output debe coincidir con las dimensiones que se colocan en los vectores de input y output de la data de entrenamiento.

## Ejemplo de una red neuronal ligeramente más compleja
En el siguiente [LINK](https://youtu.be/MYHWuuA_XcQ?t=616) muestro ejemplos en los que se pueden usar las redes neuronales, de manera visual en un archivo excel personalizado. Claro que, para redes neuronales grandes, Excel es lento y por eso es mejor Python.

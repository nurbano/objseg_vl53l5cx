# objseg_vl53l5cx
Dataset for object segmentation in indoor settings. The data was obtained with 3 ST VL53L5CX sensors.

Conjunto de datos para segementación de objetos en escenarios interiores segmentado manualmente. 
Los datos fueron obtenidos con 3 sensores ST VL53L5CX.
Se presentan 8 escenarios, cada uno con una imágen RGB y un archivo .xyz.
El archivo .xyz tiene el siguiente header:
[X, Y, Z, c, clase]

Siendo X, Y, Z las coordenas cartesianas de la nube de puntos.
c, corresponde al sensor con el acual fue obtenida la información.
Y clase, el objeto al cual corresponde. 

Se puede visualizar los datos en la app:
https://objseg-vl53l5cx.streamlit.app/

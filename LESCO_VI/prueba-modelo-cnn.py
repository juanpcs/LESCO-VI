import cv2
import mediapipe as mp
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

numeros_a_letras = {
    0: 'A',  1: 'B',  2: 'C',  3: 'D',  4: 'E',  5: 'F',  6: 'G',  7: 'H',  8: 'I',
    9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S',
    18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

# Se carga nuestro modelo entrenado desde el archivo
cnn_model = load_model('ClasificadorCNN.h5')

# Print the model summary
#cnn_model.summary()

# Inspeccionar pesos para verificar si parecen haber sido ajustados
#for layer in cnn_model.layers:
#    print(layer.name, layer.get_weights())

data = []
img_path = "mano_estandarizada_0.png"

 # Cargar la imagen usando OpenCV
img = cv2.imread(img_path)
# Normalize pixel values to the range [0, 1]
img = img / 255.0
# Añadir la imagen y su etiqueta a las listas
data.append(img)

# Convertir las listas en arrays de NumPy para su uso en ML
data = np.array(data)

print(f"Total de imágenes cargadas: {len(data)}")
print(data.shape)

prediccion_one_hot = cnn_model.predict(np.array(data))
prediccion_numero = np.argmax(prediccion_one_hot)

print(prediccion_one_hot)

print(prediccion_numero)

prediccion = numeros_a_letras[prediccion_numero]

print(prediccion)
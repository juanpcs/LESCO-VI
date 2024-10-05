import cv2
import mediapipe as mp
import os
import time
import numpy as np
from tensorflow.keras.models import load_model

# Se inicializa la cámara
cap = cv2.VideoCapture(0)

# Se inicializa mediapipe para la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence = 0.7)
mp_drawing = mp.solutions.drawing_utils

# Se define el tamaño estándar para las imágenes recortadas
standard_size = (128, 128)  

# Se carga nuestro modelo entrenado
cnn_model = load_model('ModeloEntrenado.h5')

# Print the model summary
#cnn_model.summary()

numeros_a_letras = {
    0: 'A',  1: 'B',  2: 'C',  3: 'D',  4: 'E',  5: 'F',  6: 'G',  7: 'H',  8: 'I',
    9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S',
    18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}


while True:

    data_aux = []
    # Se captura un frame de la cámara
    ret, frame = cap.read()
    
    if not ret:
        print("Error: No se pudo capturar la imagen")
        break
    
    # Se convierte la imagen de BGR a RGB para MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Se procesa la imagen para detectar la mano
    result = hands.process(image_rgb)
    
    # Se verifica si se detectó alguna mano
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            
            # Se calculan los límites de la mano (bounding box)
            h, w, _ = frame.shape
            x_min = w
            y_min = h
            x_max = y_max = 0

            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y
                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y

            # Se añade un pequeño margen alrededor de la mano
            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            # Se recorta la imagen para obtener solo la región de la mano
            cropped_hand = frame[y_min:y_max, x_min:x_max]

            # Se muestra la imagen recortada en una ventana separada
            cv2.imshow('Mano Detectada', cropped_hand)

            # Se redimensiona la imagen recortada al tamaño estándar
            resized_hand = cv2.resize(cropped_hand, standard_size)

            # Se normaliza el valor de los pixeles al rango [0, 1]
            img = resized_hand / 255.0

            # Se añade la imagen a la lista
            data_aux.append(img)

            # Se dibujan las conexiones de la mano en la imagen original
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Se realiza la predicción mediant el modelo entrenado (vector one hot)
            prediccion_one_hot = cnn_model.predict(np.array(data_aux))

            # Se castea la predicción en formato one hot a su representación númerica
            prediccion_numero = np.argmax(prediccion_one_hot)

            # Se castea la representación númerica a su correspondiente letra mediante la ayuda de un diccionario
            prediccion = numeros_a_letras[prediccion_numero]

            # Se dibuja el bounding box en la imagen
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Se define la posición del texto (10 píxeles arriba del rectángulo)
            text_position = (x_min, y_min - 10)

            # Se dibuja el texto encima del rectángulo
            cv2.putText(frame,                     # Imagen sobre la que se dibuja el texto
                        prediccion,                # Texto a mostrar
                        text_position,             # Posición (x, y) donde se coloca el texto
                        cv2.FONT_HERSHEY_SIMPLEX,  # Fuente del texto
                        1,                         # Tamaño de la fuente
                        (0, 255, 0),               # Color del texto en formato BGR
                        2,                         # Grosor del texto
                        cv2.LINE_AA)               # Tipo de línea para una mejor apariencia
    
    # Se muestra la imagen con las anotaciones en la ventana principal
    cv2.imshow('Sistema LESCO VI', frame)
    
    # Se sale del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Se libera la cámara y se cierran todas las ventanas
cap.release()
cv2.destroyAllWindows()
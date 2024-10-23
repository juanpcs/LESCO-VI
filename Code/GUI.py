import cv2
import mediapipe as mp
import os
import time
import numpy as np
from tensorflow.keras.models import load_model

# Se inicializa la cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # fijar el ancho a 640 píxeles
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # fijar el alto a 480 píxeles

# Se inicializa mediapipe para la detección de las manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence = 0.7)
mp_drawing = mp.solutions.drawing_utils

# Se define el tamaño estándar para las imágenes recortadas
standard_size = (128, 128)  

# Se carga nuestro modelo entrenado
cnn_model = load_model('ModeloEntrenado.h5')

#Diccionario utilizado para castear la predicción del modelo (número) a su respectiva letra
numeros_a_letras = {
    0: 'A',  1: 'B',  2: 'C',  3: 'D',  4: 'E',  5: 'F',  6: 'G',  7: 'H',  8: 'I',
    9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S',
    18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

# Variable de control de estados
captura_iniciada = False
ayuda_solicitada = False
mostrar_señas = False

while True:
    #Array para guardar los datos de la imagen de la seña a predecir
    data_aux = []
    # Se captura un frame de la cámara
    ret, frame = cap.read()
    
    if not ret:
        print("Error: No se pudo capturar la imagen")
        break

    # Se muestra un mensaje de espera si no ha comenzado la captura
    if not captura_iniciada and not ayuda_solicitada and not mostrar_señas:
        # Definir la posición y tamaño del rectángulo
        x_start, y_start = 65, 140  # Coordenadas del rectángulo
        box_width, box_height = 510, 185  # Tamaño del rectángulo
        
        # Se dibuja un rectángulo de fondo
        cv2.rectangle(frame, (x_start, y_start), (x_start + box_width, y_start + box_height), (255, 255, 255), -1)

        # Se dibuja el texto encima del rectángulo
        cv2.putText(frame, 
                    "Presiona 'S' para iniciar la captura", 
                    (x_start + 61, y_start + 40),  # Posición del texto dentro del rectángulo
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7,  # Tamaño de la fuente
                    (255, 0, 0),  # Color del texto (BGR)
                    1,  # Grosor del texto
                    cv2.LINE_AA)
        
        cv2.putText(frame, 
                    "Presiona 'Q' para pausar la captura", 
                    (x_start + 55, y_start + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA)

        cv2.putText(frame, 
                    "Presiona 'H' si necesitas ayuda", 
                    (x_start + 82, y_start + 120),  
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7,  
                    (255, 0, 0),  
                    1, 
                    cv2.LINE_AA)

        cv2.putText(frame, 
                    "Presiona 'L' para ver las senas reconocidas", 
                    (x_start + 10, y_start + 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA)

        # Se muestra el frame con el rectángulo y el mensaje
        cv2.imshow('Sistema LESCO VI', frame)
    
    
    if ayuda_solicitada:
        # Definir la posición y tamaño del rectángulo
        x_start, y_start = 10, 130  # Coordenadas del rectángulo
        box_width, box_height = 620, 220  # Tamaño del rectángulo 
        
        # Se dibuja un rectángulo de fondo
        cv2.rectangle(frame, (x_start, y_start), (x_start + box_width, y_start + box_height), (255, 255, 255), -1)

        # Se dibuja el texto encima del rectángulo
        cv2.putText(frame, 
                    "Instrucciones de uso:", 
                    (x_start + 5, y_start + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA)

        cv2.putText(frame, 
                    "1. Iniciar la captura de las senas presionando la tecla 'S' ", 
                    (x_start + 15, y_start + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA)

        cv2.putText(frame, 
                    "2. Mostrar a la camara la sena a reconocer, usando la mano derecha", 
                    (x_start + 15, y_start + 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA)

        cv2.putText(frame, 
                    "3. Esperar a que se muestre la traduccion", 
                    (x_start + 15, y_start + 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA)

        cv2.putText(frame, 
                    "4. Continuar realizando todas las senas deseadas", 
                    (x_start + 15, y_start + 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA)

        # Se muestra el frame con el rectángulo y el mensaje
        cv2.imshow('Sistema LESCO VI', frame)

    if mostrar_señas:
        # Se carga la imagen desde el archivo
        imagen = cv2.imread('alfabeto.png')

        if imagen is None:
            print("Error: No se pudo cargar la imagen.")
        else:
            # Definir el nuevo tamaño
            nuevo_tamano = (640, 480)
            # Redimensionar la imagen
            imagen_redimensionada = cv2.resize(imagen, nuevo_tamano)
            # Mostrar la imagen
            cv2.imshow('Sistema LESCO VI', imagen_redimensionada)
            # Esperar a que se presione una tecla para cerrar la ventana
            cv2.waitKey(0)
            mostrar_señas = False
    
    # Si la captura ha comenzado, realizar la detección de manos y predicción
    if captura_iniciada:

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
                #cv2.imshow('Mano Detectada', cropped_hand)

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
    
    # Esperar la entrada del usuario
    key = cv2.waitKey(1) & 0xFF

    # Si el usuario presiona 's', se inicia la captura
    if key == ord('s'):
        captura_iniciada = True

    # Si el usuario presiona 'q', pone en pausa la captura
    if key == ord('q'):
        captura_iniciada = False

    # Si el usuario presiona 'h', se muestran las intrucciones de uso
    if key == ord('h'):
        ayuda_solicitada = not ayuda_solicitada

    # Si el usuario presiona 'l', se muestran las señas reconocidas
    if key == ord('l'):
        mostrar_señas = not mostrar_señas

    # Verificar si la ventana ha sido cerrada
    if cv2.getWindowProperty('Sistema LESCO VI', cv2.WND_PROP_VISIBLE) < 1:
        break

# Se libera la cámara y se cierran todas las ventanas
cap.release()
cv2.destroyAllWindows()
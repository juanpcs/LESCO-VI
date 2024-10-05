import cv2
import mediapipe as mp
import os
import time

# Se inicializa la cámara
cap = cv2.VideoCapture(0)

# Se inicializa mediapipe para la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence = 0.7)
mp_drawing = mp.solutions.drawing_utils

# Se crea un directorio para guardar las fotos si no existe
output_dir = "y"
os.makedirs(output_dir, exist_ok=True)

# Contador para nombrar las imágenes
img_counter = 140

# Tiempo de inicio para controlar la captura de fotos
start_time = time.time()

# Se define el tamaño estándar para las imágenes recortadas
standard_size = (128, 128)  

while True:
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

            # Se obtiene el tiempo actual
            current_time = time.time()

            # Se verifica si han pasado x segundos desde la última captura
            if current_time - start_time >= 3:
                # Se guarda la imagen recortada
                img_name = f"mano_estandarizada_{img_counter}.png"
                cv2.imwrite(os.path.join(output_dir, img_name), resized_hand)
                print(f"Imagen estandarizada guardada como '{img_name}'")
                img_counter += 1
                
                # Se actualiza el tiempo de inicio
                start_time = current_time

            # Se dibujan las conexiones de la mano en la imagen original
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Se dibuja el bounding box en la imagen
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    # Se muestra la imagen con las anotaciones en la ventana principal
    cv2.imshow('Detección de Manos', frame)
    
    # Se sale del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# # Se libera la cámara y se cierran todas las ventanas
cap.release()
cv2.destroyAllWindows()
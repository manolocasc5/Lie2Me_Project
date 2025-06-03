import cv2
import numpy as np
import face_recognition
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array

IMG_SIZE = 224

def detectar_rostros(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")  # o "cnn" si quieres más precisión y tienes GPU
    return boxes

def extraer_region_rostro(frame, box):
    top, right, bottom, left = box
    rostro = frame[top:bottom, left:right]
    if rostro.size == 0:
        return None
    rostro = cv2.resize(rostro, (IMG_SIZE, IMG_SIZE))
    return rostro

def preprocesar_imagen(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img_to_array(img)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def predecir_emocion(modelo, imagen):
    pred = modelo.predict(imagen, verbose=0)[0][0]
    return pred

def dibujar_caja_y_texto(frame, box, texto, color=(255, 0, 0)):
    top, right, bottom, left = box
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.putText(frame, texto, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

def graficar_predicciones(predicciones, titulo="Evolución de la Predicción"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(predicciones, marker='o', linestyle='-', color='orange')
    ax.axhline(0.5, color='gray', linestyle='--', label='Umbral')
    ax.set_title(titulo)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Predicción (0=No, 1=Repetir)")
    ax.legend()
    ax.grid(True)
    return fig

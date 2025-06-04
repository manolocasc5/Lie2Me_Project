import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import ffmpeg # Asegúrate de que ffmpeg-python está instalado (pip install ffmpeg-python)
import os # Necesario para verificar si el archivo existe

# --- Configuración global ---
IMG_SIZE = 224 # Tamaño al que se redimensionan los rostros para el modelo

# Variable global para el clasificador de Haar, inicializado por una función.
face_cascade = None 

def init_face_cascade_classifier():
    """
    Inicializa el clasificador de Haar de OpenCV.
    Retorna un mensaje de estado y un booleano indicando éxito.
    """
    global face_cascade

    try:
        classifier_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(classifier_path)
        
        if face_cascade.empty():
            raise IOError(f"El clasificador de Haar NO SE CARGÓ. Archivo no encontrado o corrupto en: {classifier_path}")
        
        return "Clasificador de Haar cargado correctamente.", True
    except Exception as e:
        return f"Error crítico al cargar el clasificador de Haar: {e}. La detección de rostros no funcionará. Asegúrate de que 'haarcascade_frontalface_default.xml' está disponible y en la ruta correcta.", False

# --- FUNCIÓN: Obtener ángulo de rotación del video ---
def get_video_rotation(video_path):
    """
    Obtiene el ángulo de rotación de un video a partir de sus metadatos usando ffprobe.
    Normaliza el ángulo a 0, 90, 180, 270 grados.
    Retorna el ángulo en grados o 0 si no se encuentra rotación o hay un error.
    """
    if not os.path.exists(video_path):
        print(f"Error (get_video_rotation): El archivo de video no existe en la ruta: {video_path}")
        return 0

    try:
        # Se usa 'capture_stderr=True' para poder decodificar mensajes de error de ffmpeg
        probe = ffmpeg.probe(video_path, select_streams='v', show_streams=None) 
        
        # Busca específicamente el stream de video
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        
        if video_stream and 'tags' in video_stream:
            rotation_tag = video_stream['tags'].get('rotate')
            if rotation_tag is not None:
                try:
                    rotation = int(rotation_tag)
                    # ffprobe puede reportar 90 grados como -270 y 270 como -90.
                    # Normalizamos a 0, 90, 180, 270.
                    if rotation == -90:
                        return 270
                    elif rotation == -270:
                        return 90
                    else:
                        return rotation % 360 # Asegura que esté en el rango 0-359
                except ValueError:
                    print(f"Advertencia (get_video_rotation): El valor de rotación '{rotation_tag}' no es un número entero. Asumiendo 0 grados.")
                    return 0
        
        # Como fallback, también se puede buscar en la matriz de visualización (displaymatrix)
        # Esto es menos común para el tag 'rotate', pero a veces ocurre.
        # Es más complejo de parsear directamente el ángulo de la matriz, así que nos centramos en 'rotate'.
        # Si no se encuentra 'rotate', asumimos 0.

    except ffmpeg.Error as e:
        # Imprime el error stderr de ffmpeg para un diagnóstico más claro
        print(f"Error (get_video_rotation) al obtener metadatos de rotación con ffprobe: {e.stderr.decode()}")
    except Exception as e:
        print(f"Error inesperado (get_video_rotation) al obtener metadatos de rotación: {e}")
    
    return 0 # Si no se encuentra rotación o hay un error, asumimos 0 grados

# --- FUNCIÓN: Rotar un frame ---
def rotate_frame_if_needed(frame, rotation_angle):
    """
    Rota un frame de OpenCV según el ángulo de rotación especificado.
    
    Args:
        frame (np.array): El frame de imagen de OpenCV (BGR).
        rotation_angle (int): El ángulo de rotación en grados (0, 90, 180, 270).

    Returns:
        np.array: El frame rotado.
    """
    if rotation_angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation_angle == 270: # Equivalente a ROTATE_90_COUNTERCLOCKWISE
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return frame # No se necesita rotación o ángulo no reconocido

# --- Resto de funciones ---
def detectar_rostros(frame):
    """
    Detecta rostros en un frame dado usando el clasificador de Haar.
    Retorna una lista de bounding boxes (top, right, bottom, left).
    """
    if face_cascade is None:
        print("Advertencia: El clasificador de Haar no está inicializado. No se detectarán rostros.")
        return []
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    boxes = []
    for (x, y, w, h) in faces:
        # OpenCV devuelve (x, y, w, h). Queremos (top, right, bottom, left) para compatibilidad con el modelo.
        boxes.append((y, x + w, y + h, x)) 
    return boxes

def extraer_region_rostro(frame, box):
    """
    Extrae y redimensiona la región de un rostro de un frame.
    """
    top, right, bottom, left = box
    h_frame, w_frame, _ = frame.shape
    
    # Asegura que las coordenadas estén dentro de los límites del frame
    top = max(0, top)
    right = min(w_frame, right)
    bottom = min(h_frame, bottom)
    left = max(0, left)

    rostro = frame[top:bottom, left:right]
    
    # Valida que la región extraída no esté vacía y tenga un tamaño mínimo
    if rostro.size == 0 or rostro.shape[0] < 10 or rostro.shape[1] < 10:
        return None
    
    try:
        rostro = cv2.resize(rostro, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    except cv2.error as e:
        print(f"Error de OpenCV al redimensionar rostro: {e}")
        return None
    except Exception as e:
        print(f"Error inesperado al redimensionar rostro: {e}")
        return None
    return rostro

def preprocesar_imagen(img):
    """
    Preprocesa una imagen de rostro para el modelo de Keras.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Keras modelos suelen esperar RGB
    img = img_to_array(img)
    img = img / 255.0 # Normalización a 0-1
    return np.expand_dims(img, axis=0) # Añade dimensión de batch

def predecir_emocion(modelo, imagen_preprocesada):
    """
    Realiza una predicción de emoción usando el modelo.
    """
    try:
        pred = modelo.predict(imagen_preprocesada, verbose=0)[0][0]
        return pred
    except Exception as e:
        print(f"Error al realizar la predicción: {e}. Devolviendo 0.5 por defecto.")
        return 0.5 # Valor por defecto en caso de error

def dibujar_caja_y_texto(frame, box, texto, color=(255, 0, 0), thickness=2, font_scale=0.8):
    """
    Dibuja una bounding box y texto en un frame.
    """
    top, right, bottom, left = box
    # Asegura que las coordenadas sean enteros
    top, right, bottom, left = int(top), int(right), int(bottom), int(left)

    cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
    
    # Posiciona el texto por encima de la caja, si es posible
    text_size = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = left
    text_y = top - 10
    if text_y < text_size[1]: # Si el texto se sale por arriba, ponlo dentro de la caja
        text_y = top + text_size[1] + 5

    cv2.putText(frame, texto, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def graficar_predicciones(predicciones, titulo="Evolución de la Predicción"):
    """
    Genera una gráfica de la evolución de las predicciones.
    """
    if not predicciones:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No hay datos de predicciones para graficar.", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title(titulo)
        ax.set_xticks([]) 
        ax.set_yticks([])
        return fig

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(predicciones, marker='o', linestyle='-', color='orange', alpha=0.7)
    ax.axhline(0.5, color='gray', linestyle='--', label='Umbral (0.5)')
    ax.set_title(titulo)
    ax.set_xlabel("Número de Detección (orden cronológico)")
    ax.set_ylabel("Predicción de Predisposición (0-1)")
    ax.set_ylim(-0.1, 1.1) # Rango completo para la predicción
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout() # Ajusta el diseño para evitar superposiciones
    return fig

def guardar_video(frames, path, fps=15):
    """
    Guarda una lista de frames como un archivo de video.
    """
    if not frames:
        print("Advertencia: No hay frames para guardar en el video.")
        return False
        
    height, width, _ = frames[0].shape
    
    # Código para video MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: No se pudo abrir el archivo de salida del video: {path}. Verifica permisos o códecs instalados.")
        return False

    for i, f in enumerate(frames):
        try:
            out.write(f)
        except Exception as e:
            print(f"Error al escribir el frame {i} en el video: {e}")
            break 

    out.release()
    print(f"Video guardado exitosamente en: {path}")
    return True
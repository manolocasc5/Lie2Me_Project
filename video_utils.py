import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import os
import librosa
import soundfile as sf
import subprocess
import tensorflow_hub as hub
from sklearn.preprocessing import StandardScaler
import joblib
import math # Asegúrate de que esta línea está aquí

# --- Configuración global ---
IMG_SIZE = 224
AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION_SECONDS = 3 # Duración de cada segmento de audio para YAMNet en segundos

SCALER_PATH = "model/audio_scaler.npy"
FACE_CASCADE_PATH = "model/haarcascade_frontalface_default.xml"

# Variables globales para los recursos cargados (se inicializan al importar el módulo)
face_cascade = None
YAMNET_MODEL = None
audio_scaler = None

# --- Funciones de inicialización y carga ---

def init_face_cascade_classifier():
    """Inicializa el clasificador de Haar para detección facial."""
    global face_cascade
    if face_cascade is None: # Solo intenta cargar si no está ya cargado
        if not os.path.exists(FACE_CASCADE_PATH):
            return f"Error: Clasificador de Haar no encontrado en '{FACE_CASCADE_PATH}'.", False
        try:
            face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
            if face_cascade.empty():
                return "Error: Clasificador de Haar no pudo ser cargado. Archivo corrupto o inválido.", False
        except Exception as e:
            return f"Error al cargar clasificador de Haar: {e}", False
    return "Clasificador de Haar inicializado.", True

# Carga global de YAMNET y el escalador de audio al importar el módulo
# Estas cargas ocurren una vez cuando video_utils se importa por primera vez
try:
    print("Cargando modelo YAMNet para preprocesamiento de audio...")
    # Puedes descargar el modelo de TF Hub o usar una copia local si lo has guardado
    # YAMNET_MODEL_HANDLE = 'https://tfhub.dev/google/yamnet/1' # Versión online
    YAMNET_MODEL_HANDLE = 'model/yamnet' # Versión local (asegúrate de tener la carpeta 'yamnet' dentro de 'model')
    YAMNET_MODEL = hub.load(YAMNET_MODEL_HANDLE)
    print("Modelo YAMNet cargado correctamente.")
except Exception as e:
    print(f"Error al cargar YAMNet: {e}. La predicción de audio podría fallar.")
    YAMNET_MODEL = None

try:
    if os.path.exists(SCALER_PATH):
        audio_scaler = joblib.load(SCALER_PATH)
        print("Escalador de audio cargado exitosamente.")
    else:
        print(f"Advertencia: El archivo del escalador de audio no se encontró en '{SCALER_PATH}'. Las predicciones de audio serán imprecisas o fallarán.")
        audio_scaler = None # Asegurarse de que sea None si no se carga
except Exception as e:
    print(f"Error al cargar el escalador de audio: {e}. La predicción de audio podría ser imprecisa o fallar.")
    audio_scaler = None


# --- Funciones de preprocesamiento y predicción ---

def preprocesar_y_predecir_video(frame, model_video):
    """
    Preprocesar un frame de vídeo para la detección facial y predecir la emoción.
    Devuelve la probabilidad de predisposición y el frame con la detección dibujada.
    """
    # Asegúrate de que el clasificador de Haar esté inicializado.
    # Si no, intenta inicializarlo o retorna sin procesamiento facial.
    if face_cascade is None:
        # Intenta inicializarlo de nuevo por si falló la primera vez
        status, success = init_face_cascade_classifier()
        if not success:
            # print(f"Clasificador de Haar no inicializado: {status}. No se detectarán caras.")
            return None, frame # Retornar el frame original si no hay clasificador

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    prediccion_frame = None
    frame_con_deteccion = frame.copy()

    if len(faces) > 0:
        (x, y, w, h) = faces[0] # Tomar la primera cara detectada

        face_roi = frame[y:y+h, x:x+w]
        # Asegurarse de que el ROI tiene dimensiones válidas antes de redimensionar
        if face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
            return None, frame_con_deteccion

        face_roi = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
        face_roi = img_to_array(face_roi)
        face_roi = np.expand_dims(face_roi, axis=0) # Añadir dimensión de batch
        face_roi = face_roi / 255.0 # Normalizar

        if model_video:
            prediccion = model_video.predict(face_roi, verbose=0)[0][0]
            prediccion_frame = float(prediccion)

            # Dibujar rectángulo y etiqueta
            color = (0, 255, 0) if prediccion_frame > 0.5 else (0, 0, 255) # Verde para predisposición, Rojo para no
            label = f"Predisposicion: {prediccion_frame:.2f}" if prediccion_frame > 0.5 else f"No Predisposicion: {prediccion_frame:.2f}"
            cv2.rectangle(frame_con_deteccion, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame_con_deteccion, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        # else:
            # print("Advertencia: Modelo de vídeo no cargado. No se realizarán predicciones faciales.")

    return prediccion_frame, frame_con_deteccion


def preprocesar_audio_for_yamnet(audio_waveform, sr=AUDIO_SAMPLE_RATE, duration=AUDIO_DURATION_SECONDS):
    """
    Preprocesa una forma de onda de audio para que sea compatible con YAMNet,
    ajustando la duración y normalizando.
    """
    if audio_waveform is None or len(audio_waveform) == 0:
        # print("Advertencia: Forma de onda de audio vacía o nula.")
        return None

    # Normalizar la amplitud de la forma de onda
    max_abs_val = np.max(np.abs(audio_waveform))
    if max_abs_val > 0:
        y = audio_waveform / max_abs_val
    else:
        y = audio_waveform # Evitar división por cero, aunque esto indica un audio silencioso

    # Asegurar la duración correcta
    target_length = sr * duration
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    elif len(y) > target_length:
        y = y[:target_length]

    return y.astype(np.float32)


def extract_yamnet_embeddings(audio_waveform, yamnet_model):
    """
    Extrae los embeddings de YAMNet de una forma de onda de audio.
    """
    if audio_waveform is None or yamnet_model is None:
        # print("Advertencia: Forma de onda o modelo YAMNet nulos para extracción de embeddings.")
        return None

    # YAMNet devuelve scores, embeddings y un espectrograma
    # Asegúrate de que el input para YAMNet sea una forma de onda de audio de tipo tf.float32
    # YAMNet espera un tensor de 1D
    audio_tensor = tf.constant(audio_waveform, dtype=tf.float32)

    # Si YAMNet espera un batch, expandir dimensiones (aunque generalmente no para yamnet/1)
    # Si la forma de onda ya es de la longitud esperada por YAMNet, no hay problema.
    # scores, embeddings, spectrogram = yamnet_model(audio_tensor) # Esto funcionaría si audio_waveform ya tiene la longitud adecuada

    # YAMNet procesa internamente en segmentos, la salida 'embeddings' ya es el promedio para el clip
    # Para yamnet/1, la entrada es de 1 segundo a 16kHz
    # Si tu segmento es de AUDIO_DURATION_SECONDS (ej. 3s), YAMNet lo procesa y devuelve un embedding por segundo.
    # Por eso, `embeddings.numpy().mean(axis=0)` es adecuado.

    try:
        scores, embeddings, spectrogram = yamnet_model(audio_tensor)
        if embeddings.shape[0] > 0:
            # Si YAMNet produce múltiples embeddings (para segmentos internos de 1s), promedia
            return np.mean(embeddings.numpy(), axis=0)
        else:
            # print("Advertencia: YAMNet no produjo embeddings para el segmento de audio.")
            return None
    except Exception as e:
        print(f"Error al ejecutar YAMNet: {e}. El segmento de audio podría ser inválido.")
        return None


def preprocesar_y_predecir_audio(audio_path, model_audio, yamnet_model, scaler_audio):
    """
    Carga un archivo de audio, lo preprocesa con YAMNet, lo escala y predice la emoción.
    Devuelve una lista de probabilidades de predisposición para cada segmento.
    """
    if not os.path.exists(audio_path):
        print(f"Advertencia: Archivo de audio no encontrado en {audio_path}")
        return None

    if yamnet_model is None or scaler_audio is None or model_audio is None:
        print("Advertencia: Modelos o escalador de audio no cargados. Saltando predicción de audio.")
        return None

    all_segment_embeddings = []
    try:
        # Cargar el audio. Librosa lo resamplea automáticamente al sr especificado.
        y, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"Error al cargar archivo de audio con librosa: {e}. Puede que el archivo esté corrupto o sea incompatible.")
        return None

    # Si el audio es muy corto, no podemos segmentar
    if len(y) < sr * AUDIO_DURATION_SECONDS:
        # print(f"Advertencia: El audio es demasiado corto ({len(y)/sr:.2f}s) para ser segmentado. Necesita al menos {AUDIO_DURATION_SECONDS} segundos.")
        # Intentar procesarlo como un solo segmento si tiene sentido
        processed_waveform = preprocesar_audio_for_yamnet(y, sr=sr, duration=AUDIO_DURATION_SECONDS)
        if processed_waveform is not None:
            embedding = extract_yamnet_embeddings(processed_waveform, yamnet_model)
            if embedding is not None:
                all_segment_embeddings.append(embedding)
    else:
        # Segmentar el audio si es más largo que la duración definida
        segment_length_samples = sr * AUDIO_DURATION_SECONDS
        num_segments = math.ceil(len(y) / segment_length_samples)

        for i in range(num_segments):
            start_sample = i * segment_length_samples
            end_sample = min((i + 1) * segment_length_samples, len(y))
            segment_waveform = y[start_sample:end_sample]

            # Asegurarse de que el segmento tenga la longitud esperada por YAMNet
            processed_waveform = preprocesar_audio_for_yamnet(segment_waveform, sr=sr, duration=AUDIO_DURATION_SECONDS)
            if processed_waveform is not None:
                embedding = extract_yamnet_embeddings(processed_waveform, yamnet_model)
                if embedding is not None:
                    all_segment_embeddings.append(embedding)

    if not all_segment_embeddings:
        print("No se pudieron extraer embeddings de audio. No hay datos para predecir.")
        return None

    X_embeddings = np.array(all_segment_embeddings)

    if scaler_audio:
        # Asegurarse de que el escalador tiene la misma cantidad de features que los embeddings
        if X_embeddings.shape[1] != scaler_audio.n_features_in_:
            print(f"Error: La dimensión de los embeddings ({X_embeddings.shape[1]}) no coincide con el escalador ({scaler_audio.n_features_in_}).")
            return None
        X_scaled = scaler_audio.transform(X_embeddings)
    else:
        print("Advertencia: Escalador de audio no disponible. Las predicciones pueden ser imprecisas.")
        X_scaled = X_embeddings # Procede sin escalar si no hay escalador

    predicciones = model_audio.predict(X_scaled, verbose=0)

    # Asegurarse de que la salida es una lista simple de floats
    return predicciones.flatten().tolist()


def fusionar_predicciones(predicciones_video, predicciones_audio, video_weight, audio_weight):
    """
    Fusiona las predicciones de vídeo y audio.
    Normaliza las listas para que tengan la misma longitud antes de fusionar.
    Devuelve la lista fusionada, y las listas de vídeo y audio remuestreadas.
    """
    # Si no hay ninguna predicción, devuelve listas vacías para todo
    if not predicciones_video and not predicciones_audio:
        return [], [], []

    # Inicializa las listas remuestreadas.
    predicciones_video_resampled = []
    predicciones_audio_resampled = []

    # Casos donde solo hay vídeo o solo hay audio (para asegurar consistencia en el retorno)
    if not predicciones_audio:
        # Si no hay audio, la fusión es solo el vídeo.
        # Devuelve el vídeo original, una lista vacía para audio (o replicada para gráfico), y el vídeo fusionado.
        # Asegurarse de que si el audio_weight es 0, la fusión sea solo el vídeo.
        if audio_weight == 0:
            return predicciones_video, predicciones_video, []
        else: # Si hay peso de audio pero no hay datos de audio, la fusión es solo el video
            return predicciones_video, predicciones_video, [] # Mantener predicciones_audio_resampled vacía si no hay audio
    if not predicciones_video:
        # Si no hay vídeo, la fusión es solo el audio.
        if video_weight == 0:
            return predicciones_audio, [], predicciones_audio
        else: # Si hay peso de vídeo pero no hay datos de vídeo, la fusión es solo el audio
            return predicciones_audio, [], predicciones_audio


    len_video = len(predicciones_video)
    len_audio = len(predicciones_audio)

    # Remuestreo para igualar longitudes
    if len_video > len_audio:
        predicciones_audio_resampled = np.interp(
            np.linspace(0, len_audio - 1, len_video),
            np.arange(len_audio),
            predicciones_audio
        ).tolist()
        predicciones_video_resampled = predicciones_video # El vídeo no necesita remuestreo
    elif len_audio > len_video:
        predicciones_video_resampled = np.interp(
            np.linspace(0, len_video - 1, len_audio),
            np.arange(len_video),
            predicciones_video
        ).tolist()
        predicciones_audio_resampled = predicciones_audio # El audio no necesita remuestreo
    else: # Misma longitud
        predicciones_video_resampled = predicciones_video
        predicciones_audio_resampled = predicciones_audio

    # Realizar la fusión
    fusionadas = []
    # Asegúrate de que las listas tienen la misma longitud antes del zip para evitar IndexError
    min_len = min(len(predicciones_video_resampled), len(predicciones_audio_resampled))
    for pv, pa in zip(predicciones_video_resampled[:min_len], predicciones_audio_resampled[:min_len]):
        fusionadas.append((pv * video_weight) + (pa * audio_weight))

    return fusionadas, predicciones_video_resampled, predicciones_audio_resampled


# --- Función para extraer audio del vídeo usando FFmpeg ---
def extraer_audio_ffmpeg(video_path, output_audio_path):
    """
    Extrae la pista de audio de un archivo de vídeo usando FFmpeg.
    """
    command = [
        'ffmpeg',
        '-y', # Sobreescribir el archivo de salida si existe, sin preguntar
        '-i', video_path,
        '-vn',             # Deshabilita el vídeo
        '-acodec', 'pcm_s16le', # Codec de audio: PCM de 16 bits little-endian
        '-ar', str(AUDIO_SAMPLE_RATE), # Frecuencia de muestreo (16000 Hz)
        '-ac', '1',        # Un solo canal de audio (mono)
        output_audio_path
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        # print("FFmpeg STDOUT (extraer audio):", result.stdout)
        # print("FFmpeg STDERR (extraer audio):", result.stderr)
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg falló al extraer audio (código {e.returncode}).\n"
        error_msg += f"Comando: {' '.join(e.cmd)}\n"
        error_msg += f"STDOUT: {e.stdout}\n"
        error_msg += f"STDERR: {e.stderr}"
        # print(error_msg)
        raise RuntimeError(f"FFmpeg falló al extraer audio: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg no encontrado. Asegúrate de que FFmpeg está instalado y accesible en tu PATH.")
    except Exception as e:
        raise RuntimeError(f"Un error inesperado ocurrió al extraer audio: {e}")

# --- Función para guardar frames como vídeo (requiere FFmpeg) ---
def guardar_frames_como_video(frames, output_path, fps):
    """
    Guarda una lista de frames (imágenes NumPy) como un archivo de vídeo MP4 usando FFmpeg.
    """
    if not frames:
        print("No hay frames para guardar.")
        return

    height, width, _ = frames[0].shape
    
    # FFmpeg comando para crear un MP4 a partir de frames raw
    command = [
        'ffmpeg',
        '-y', # Sobreescribir el archivo de salida si existe, sin preguntar
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f"{width}x{height}",
        '-pix_fmt', 'bgr24', # Formato de píxel de OpenCV (BGR)
        '-r', str(fps),     # Framerate
        '-i', '-',          # Entrada desde stdin
        '-c:v', 'libx264',  # Codec de vídeo
        '-pix_fmt', 'yuv420p', # Formato de píxel de salida (compatible con la mayoría de reproductores)
        '-preset', 'fast',  # Velocidad de codificación (ultra-fast, superfast, fast, medium, slow, etc.)
        output_path
    ]

    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        for frame in frames:
            # Escribir los bytes del frame directamente al stdin de FFmpeg
            process.stdin.write(frame.tobytes())

        process.stdin.close() # Importante cerrar stdin para que FFmpeg sepa que no hay más datos

        # Capturar la salida de error para depuración
        stderr = process.stderr.read().decode()
        process.wait() # Esperar a que el proceso termine

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command, stderr=stderr)

    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg falló al guardar vídeo (código {e.returncode}).\n"
        error_msg += f"Comando: {' '.join(e.cmd)}\n"
        error_msg += f"STDERR: {e.stderr}"
        # print(error_msg)
        raise RuntimeError(f"FFmpeg falló al guardar vídeo: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg no encontrado. Asegúrate de que FFmpeg está instalado y accesible en tu PATH.")
    except Exception as e:
        raise RuntimeError(f"Un error inesperado ocurrió al guardar vídeo: {e}")
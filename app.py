import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import tempfile
from datetime import datetime
import pandas as pd
import math
import matplotlib.pyplot as plt
import subprocess # Necesario para la comprobación de FFmpeg

# Importa el módulo video_utils
# ESTA ES LA PRIMERA Y ÚNICA IMPORTACIÓN DE video_utils.py EN app.py
import video_utils as vu

# --- Configuración de la página de Streamlit ---
st.set_page_config(layout="wide", page_title="Lie2Me - Detección de Predisposición")

st.title("Lie2Me - Detección de Predisposición (Vídeo + Audio)")

# Carga de modelos y clasificadores
# Se usa st.cache_resource para cargar los modelos solo una vez
# ESTA FUNCIÓN load_resources() DEBE ESTAR EN app.py, NO EN video_utils.py
@st.cache_resource
def load_resources():
    model_video = None
    model_audio = None

    try:
        model_video = tf.keras.models.load_model("model/mobilenetv2_emotion_binario_finetune.h5")
        st.success("✅ Modelo de vídeo cargado correctamente.")
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo de vídeo: {e}. Asegúrate de que 'model/mobilenetv2_emotion_binario_finetune.h5' existe y es válido.")
        model_video = None

    try:
        model_audio = tf.keras.models.load_model("model/audio_emotion_model.h5")
        st.success("✅ Modelo de audio cargado correctamente.")
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo de audio: {e}. Asegúrate de que 'model/audio_emotion_model.h5' existe y es válido.")
        model_audio = None
    
    # Inicializar el clasificador de Haar usando la función de video_utils
    status_haar, success_haar = vu.init_face_cascade_classifier()
    if success_haar:
        st.success(f"✅ {status_haar}")
    else:
        st.error(f"❌ {status_haar}")

    # YAMNet y el escalador de audio se inicializan globalmente en video_utils
    # Aquí solo verificamos si se cargaron correctamente al importar vu
    if vu.YAMNET_MODEL is None or vu.audio_scaler is None:
        st.warning("⚠️ El modelo YAMNet o el escalador de audio no se cargaron correctamente. El análisis de audio puede ser inexacto o no realizarse.")
    else:
        st.success("✅ YAMNet y escalador de audio cargados.")

    return model_video, model_audio

# Cargar los recursos al inicio de la aplicación
model_video, model_audio = load_resources()

# Inicializar estados de sesión si no existen
if 'predicciones_video_list' not in st.session_state:
    st.session_state.predicciones_video_list = []
if 'predicciones_audio_list' not in st.session_state:
    st.session_state.predicciones_audio_list = []
if 'frames_procesados_para_guardar' not in st.session_state:
    st.session_state.frames_procesados_para_guardar = []
if 'fps_video_original' not in st.session_state:
    st.session_state.fps_video_original = 30 # Valor por defecto
if 'predicciones_finales_fusionadas' not in st.session_state:
    st.session_state.predicciones_finales_fusionadas = []
if 'predicciones_video_resampled' not in st.session_state:
    st.session_state.predicciones_video_resampled = []
if 'predicciones_audio_resampled' not in st.session_state:
    st.session_state.predicciones_audio_resampled = []
if 'temp_audio_path' not in st.session_state:
    st.session_state.temp_audio_path = None
if 'model_audio_loaded' not in st.session_state:
    # Esta bandera indica si el análisis de audio es *potencialmente* posible
    st.session_state.model_audio_loaded = (model_audio is not None and vu.YAMNET_MODEL is not None and vu.audio_scaler is not None)


# Contenedor para el frame de vídeo o mensaje de procesamiento
stframe = st.empty()

# Opciones de fuente de vídeo en la barra lateral
st.sidebar.header("Configuración de Entrada")
fuente = st.sidebar.radio("Selecciona la fuente de vídeo:", ("Cámara en vivo", "Subir vídeo"), index=1)

# Sliders para ponderar las predicciones
st.sidebar.header("Ponderación de Predicciones")
video_weight = st.sidebar.slider("Peso de la Predicción por Vídeo", 0.0, 1.0, 0.7, 0.05)
audio_weight = st.sidebar.slider("Peso de la Predicción por Audio", 0.0, 1.0, 0.3, 0.05)

total_weight = video_weight + audio_weight
if total_weight > 0:
    video_weight /= total_weight
    audio_weight /= total_weight
else:
    # Asegurar valores por defecto si la suma es 0
    video_weight = 0.5
    audio_weight = 0.5

st.sidebar.write(f"Pesos normalizados: Vídeo={video_weight:.2f}, Audio={audio_weight:.2f}")

# --- Lógica principal basada en la fuente de vídeo ---
if fuente == "Cámara en vivo":
    st.warning("⚠️ La funcionalidad de cámara en vivo está en desarrollo y podría no funcionar en todos los entornos de navegador/servidor.")
    st.info("💡 Para capturar la cámara, asegúrate de que tu navegador tiene permisos de acceso a la cámara.")
    st.info("❗ Nota: El análisis de audio no está disponible para la cámara en vivo en esta versión.")

    run_camera = st.checkbox("▶️ Activar Cámara", key="camera_activation_checkbox")

    cap = None
    if run_camera:
        if not model_video:
            st.error("❌ El modelo de vídeo no se cargó. No se puede realizar el análisis de vídeo en vivo.")
            # Desactivar el checkbox de la cámara automáticamente si el modelo no está listo
            st.session_state.camera_activation_checkbox = False 
            st.stop() # Detiene la ejecución aquí si el modelo no está listo

        try:
            cap = cv2.VideoCapture(0) # Intenta abrir la cámara
            if not cap.isOpened():
                st.error("❌ No se pudo acceder a la cámara. Asegúrate de que no esté en uso por otra aplicación y de que los permisos son correctos.")
                cap = None
        except Exception as e:
            st.error(f"❌ Error al inicializar la cámara: {e}")
            cap = None

        if cap:
            st.session_state.fps_video_original = cap.get(cv2.CAP_PROP_FPS)
            if st.session_state.fps_video_original <= 0:
                st.session_state.fps_video_original = 30 # Fallback si no se obtiene un FPS válido

            # Reiniciar listas para nueva ejecución
            st.session_state.predicciones_video_list = []
            st.session_state.predicciones_audio_list = [] # Siempre vacía para cámara en vivo
            st.session_state.frames_procesados_para_guardar = []
            st.session_state.predicciones_finales_fusionadas = []
            st.session_state.predicciones_video_resampled = []
            st.session_state.predicciones_audio_resampled = []
            st.session_state.temp_audio_path = None

            frame_count = 0
            # --- PARÁMETROS PARA LA CÁMARA EN VIVO ---
            CAPTURA_SEGUNDOS_CAM = 10 # Define cuántos segundos quieres capturar
            MAX_FRAMES_CAPTURA_CAM = int(st.session_state.fps_video_original * CAPTURA_SEGUNDOS_CAM)
            
            st_camera_status = st.empty()
            st_camera_status.info(f"⏳ Capturando y procesando vídeo de la cámara en vivo por {CAPTURA_SEGUNDOS_CAM} segundos...")

            try:
                # Bucle de captura con límite de frames
                while run_camera and frame_count < MAX_FRAMES_CAPTURA_CAM:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("⚠️ No se pudieron leer más frames de la cámara.")
                        break
                    
                    # Puedes ajustar display_width a tu preferencia (ej. 640, 480, 320).
                    # Cuanto menor sea, más rápido y pequeño se verá.
                    display_width = 640 # Ancho deseado para la visualización y procesamiento
                    
                    # Calcula la altura manteniendo la relación de aspecto original
                    # Asegúrate de que frame.shape[1] (ancho original) no sea cero para evitar división por cero
                    if frame.shape[1] == 0:
                        st.warning("El ancho del frame de la cámara es cero. Omitiendo redimensionamiento.")
                        resized_frame = frame
                    else:
                        display_height = int(frame.shape[0] * (display_width / frame.shape[1]))
                        # Asegurarse de que las dimensiones resultantes sean válidas (positivas)
                        if display_width <= 0 or display_height <= 0:
                            st.warning("Dimensiones de frame inválidas después de redimensionar. Omitiendo frame.")
                            continue # Saltar al siguiente frame
                        resized_frame = cv2.resize(frame, (display_width, display_height))
                    
                    
                    prediccion_video, frame_con_deteccion = vu.preprocesar_y_predecir_video(resized_frame, model_video)
                    if prediccion_video is not None:
                        st.session_state.predicciones_video_list.append(prediccion_video)
                        st.session_state.frames_procesados_para_guardar.append(frame_con_deteccion)

                    stframe.image(frame_con_deteccion, channels="BGR")
                    frame_count += 1
                    st_camera_status.text(f"📸 Capturando... Frame {frame_count}/{MAX_FRAMES_CAPTURA_CAM}")

            finally:
                if cap:
                    cap.release()
                st_camera_status.empty() # Limpia el mensaje de captura
                st.info("✅ Cámara detenida. Calculando resultados de vídeo...")
                
                if st.session_state.predicciones_video_list:
                    # Para la cámara en vivo, las predicciones de audio son una lista vacía
                    (st.session_state.predicciones_finales_fusionadas,
                     st.session_state.predicciones_video_resampled,
                     st.session_state.predicciones_audio_resampled) = vu.fusionar_predicciones(
                        st.session_state.predicciones_video_list,
                        [], # Se pasa una lista vacía para audio
                        video_weight,
                        audio_weight
                    )
                    st.success("✅ Análisis completado para vídeo en vivo!")
                else:
                    st.warning("⚠️ No se capturaron suficientes frames o no se detectaron caras para el análisis de vídeo.")
        else:
            st.info("Por favor, haz clic en '▶️ Activar Cámara' para iniciar la captura.")


elif fuente == "Subir vídeo":
    uploaded_file = st.file_uploader("📂 Sube un archivo de vídeo", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        if not model_video:
            st.error("❌ El modelo de vídeo no se cargó. No se puede realizar el análisis de vídeo.")
            st.stop()
        if not st.session_state.model_audio_loaded and audio_weight > 0:
            st.warning("⚠️ El modelo de audio o sus dependencias no se cargaron correctamente. Las predicciones se basarán solo en el vídeo.")

        # Reiniciar listas para nueva ejecución
        st.session_state.predicciones_video_list = []
        st.session_state.predicciones_audio_list = []
        st.session_state.frames_procesados_para_guardar = []
        st.session_state.predicciones_finales_fusionadas = []
        st.session_state.predicciones_video_resampled = []
        st.session_state.predicciones_audio_resampled = []
        st.session_state.temp_audio_path = None

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
            temp_video_file.write(uploaded_file.read())
            temp_video_path = temp_video_file.name

        cap = cv2.VideoCapture(temp_video_path)

        if not cap.isOpened():
            st.error("❌ Error al abrir el archivo de vídeo. Asegúrate de que es un formato compatible y no está corrupto.")
            os.unlink(temp_video_path)
            st.stop()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.session_state.fps_video_original = cap.get(cv2.CAP_PROP_FPS)
        if st.session_state.fps_video_original <= 0: # Fallback
            st.session_state.fps_video_original = 30

        progress_bar = st.progress(0)
        st_status_text = st.empty()

        # Paso de extracción de audio
        st_status_text.info("🎵 Extrayendo audio del vídeo... (Esto puede tardar unos segundos)")
        st.session_state.temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
        try:
            vu.extraer_audio_ffmpeg(temp_video_path, st.session_state.temp_audio_path)
            st_status_text.success("✅ Audio extraído correctamente.")
        except Exception as e:
            st_status_text.error(f"❌ Error al extraer audio: {e}. Asegúrate de que FFmpeg está instalado y accesible en tu PATH. El análisis de audio será omitido.")
            if os.path.exists(st.session_state.temp_audio_path):
                os.unlink(st.session_state.temp_audio_path)
            st.session_state.temp_audio_path = None # Indicar que no hay audio temporal
            st.session_state.predicciones_audio_list = [] # Asegurarse de que esté vacía


        # Paso de procesamiento de vídeo
        st_status_text.info("🎥 Procesando frames de vídeo para detección facial y emocional...")
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            prediccion_video, frame_con_deteccion = vu.preprocesar_y_predecir_video(frame, model_video)
            if prediccion_video is not None:
                st.session_state.predicciones_video_list.append(prediccion_video)
                st.session_state.frames_procesados_para_guardar.append(frame_con_deteccion)

            frame_count += 1
            progress = min(int((frame_count / total_frames) * 100), 100)
            progress_bar.progress(progress)
            st_status_text.text(f"Procesando frame {frame_count}/{total_frames} (Vídeo)...")

        cap.release()
        os.unlink(temp_video_path)
        st_status_text.success("✅ Análisis de vídeo completado.")

        # Paso de procesamiento de audio (solo si el modelo y el audio temporal están disponibles)
        if st.session_state.model_audio_loaded and st.session_state.temp_audio_path and os.path.exists(st.session_state.temp_audio_path):
            st_status_text.info("🎙️ Analizando audio del vídeo...")
            st.session_state.predicciones_audio_list = vu.preprocesar_y_predecir_audio(st.session_state.temp_audio_path, model_audio, vu.YAMNET_MODEL, vu.audio_scaler)
            if st.session_state.predicciones_audio_list is None or not st.session_state.predicciones_audio_list:
                st.session_state.predicciones_audio_list = [] # Asegurar que es una lista vacía si falla
                st.warning("⚠️ No se pudieron obtener predicciones de audio válidas (audio demasiado corto o problema en el procesamiento).")
            else:
                st.success("✅ Análisis de audio completado.")
        else:
            st.info("ℹ️ Análisis de audio omitido (modelo de audio no cargado o archivo de audio no disponible/válido).")
            st.session_state.predicciones_audio_list = [] # Asegurar que esté vacía si se omite


        # Fusión de resultados
        st_status_text.info("✨ Fusionando resultados de vídeo y audio...")
        (st.session_state.predicciones_finales_fusionadas,
         st.session_state.predicciones_video_resampled,
         st.session_state.predicciones_audio_resampled) = vu.fusionar_predicciones(
            st.session_state.predicciones_video_list,
            st.session_state.predicciones_audio_list,
            video_weight,
            audio_weight
        )
        progress_bar.empty()
        st_status_text.empty()
        st.success("✅ Análisis completado!")

        # Limpiar archivo de audio temporal
        if st.session_state.temp_audio_path and os.path.exists(st.session_state.temp_audio_path):
            os.unlink(st.session_state.temp_audio_path)
            st.session_state.temp_audio_path = None


# --- Mostrar resultados y gráficos si hay predicciones ---
def mostrar_resultados():
    predicciones_finales_fusionadas = st.session_state.predicciones_finales_fusionadas
    predicciones_video_resampled = st.session_state.predicciones_video_resampled
    predicciones_audio_resampled = st.session_state.predicciones_audio_resampled

    if predicciones_finales_fusionadas:
        st.subheader("📊 Resultados del Análisis de Predisposición")

        # Predicción general fusionada
        prediccion_promedio_fusionada = np.mean(predicciones_finales_fusionadas) if predicciones_finales_fusionadas else 0
        col1, col2 = st.columns([1, 2]) # Ajustar proporciones de columna
        with col1:
            st.markdown("#### Predicción Global (Fusión)")
            if prediccion_promedio_fusionada > 0.5:
                st.success(f"**Predisposición Detectada:** {prediccion_promedio_fusionada:.2f}")
                st.balloons()
            else:
                st.error(f"**No Predisposición Detectada:** {prediccion_promedio_fusionada:.2f}")

        # Gráfico de evolución temporal de la predisposición (Fusión)
        with col2:
            st.markdown("#### Evolución de la Probabilidad de Predisposición (Fusión)")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(predicciones_finales_fusionadas, label='Probabilidad (Fusión)', color='purple', linewidth=2)
            ax.axhline(0.5, color='r', linestyle='--', label='Umbral (0.5)')
            ax.set_xlabel('Segmento de Tiempo')
            ax.set_ylabel('Probabilidad')
            ax.set_title('Evolución Temporal de la Predisposición (Fusión)')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        
        st.markdown("---") # Separador visual

        # Sección para predicciones de VÍDEO
        st.subheader("Componente de Análisis por Vídeo")
        if predicciones_video_resampled:
            prediccion_promedio_video = np.mean(st.session_state.predicciones_video_list) if st.session_state.predicciones_video_list else 0
            st.info(f"Probabilidad de predisposición promedio por **Vídeo**: **{prediccion_promedio_video:.2f}**")
            
            fig_video, ax_video = plt.subplots(figsize=(10, 3))
            ax_video.plot(predicciones_video_resampled, label='Probabilidad (Vídeo)', color='blue')
            ax_video.axhline(0.5, color='r', linestyle='--', label='Umbral (0.5)')
            ax_video.set_xlabel('Segmento de Tiempo')
            ax_video.set_ylabel('Probabilidad')
            ax_video.set_title('Evolución Temporal de la Predisposición (Vídeo)')
            ax_video.legend()
            ax_video.grid(True)
            st.pyplot(fig_video)
        else:
            st.warning("⚠️ No se obtuvieron predicciones de vídeo (posiblemente no se detectaron caras o el vídeo es muy corto).")


        # Sección para predicciones de AUDIO
        st.subheader("Componente de Análisis por Audio")
        if st.session_state.model_audio_loaded and predicciones_audio_resampled:
            prediccion_promedio_audio = np.mean(st.session_state.predicciones_audio_list) if st.session_state.predicciones_audio_list else 0
            st.info(f"Probabilidad de predisposición promedio por **Audio**: **{prediccion_promedio_audio:.2f}**")

            fig_audio, ax_audio = plt.subplots(figsize=(10, 3))
            ax_audio.plot(predicciones_audio_resampled, label='Probabilidad (Audio)', color='green')
            ax_audio.axhline(0.5, color='r', linestyle='--', label='Umbral (0.5)')
            ax_audio.set_xlabel('Segmento de Tiempo')
            ax_audio.set_ylabel('Probabilidad')
            ax_audio.set_title('Evolución Temporal de la Predisposición (Audio)')
            ax_audio.legend()
            ax_audio.grid(True)
            st.pyplot(fig_audio)
        else:
            if not st.session_state.model_audio_loaded:
                st.error("❌ El modelo de audio o sus dependencias no se pudieron cargar. No se realizaron predicciones de audio.")
            else:
                st.warning("⚠️ No se obtuvieron predicciones de audio válidas (posiblemente el audio es muy corto o el formato no es compatible).")
        
        st.markdown("---") # Separador visual

        # Gráfico Comparativo Final
        if predicciones_video_resampled or predicciones_audio_resampled:
            st.subheader("📈 Comparativa de Predicciones (Vídeo, Audio y Fusión)")
            fig_comp, ax_comp = plt.subplots(figsize=(12, 6))
            
            if predicciones_video_resampled:
                ax_comp.plot(predicciones_video_resampled, label='Predicción Vídeo (Remuestreada)', color='blue', alpha=0.7)
            if predicciones_audio_resampled:
                ax_comp.plot(predicciones_audio_resampled, label='Predicción Audio (Remuestreada)', color='green', alpha=0.7)
            
            ax_comp.plot(predicciones_finales_fusionadas, label='Predicción Fusionada', color='purple', linewidth=2)
            ax_comp.axhline(0.5, color='r', linestyle='--', label='Umbral (0.5)')
            ax_comp.set_xlabel('Segmento de Tiempo (Base de remuestreo)')
            ax_comp.set_ylabel('Probabilidad')
            ax_comp.set_title('Comparativa de Predicciones de Predisposición')
            ax_comp.legend()
            ax_comp.grid(True)
            st.pyplot(fig_comp)


        # Ofrecer guardar el vídeo procesado con detecciones faciales
        if st.session_state.frames_procesados_para_guardar:
            st.markdown("---") # Separador visual
            st.subheader("Guardar Vídeo Procesado")
            if st.button("⬇️ Guardar y Descargar Vídeo Procesado"):
                output_video_filename = f"video_procesado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

                try:
                    with st.spinner("⏳ Generando vídeo procesado... Esto puede tardar unos minutos para vídeos largos."):
                        vu.guardar_frames_como_video(st.session_state.frames_procesados_para_guardar, temp_output_path, int(st.session_state.fps_video_original))
                    
                    with open(temp_output_path, "rb") as file:
                        st.download_button(
                            label="Haz clic para descargar el vídeo",
                            data=file.read(),
                            file_name=output_video_filename,
                            mime="video/mp4"
                        )
                    st.success(f"✅ Vídeo guardado exitosamente en: {output_video_filename}")
                    os.unlink(temp_output_path) # Limpiar archivo temporal
                except Exception as e:
                    st.error(f"❌ Error al guardar o descargar el vídeo procesado: {e}")
                    st.warning("Asegúrate de que FFmpeg está instalado correctamente en tu sistema para el guardado de vídeo.")

        # Limpiar las listas de predicciones y frames guardados para la siguiente ejecución
        # Esto es crucial para que los resultados no persistan si el usuario sube otro vídeo
        st.session_state.predicciones_video_list = []
        st.session_state.predicciones_audio_list = []
        st.session_state.frames_procesados_para_guardar = []
        st.session_state.predicciones_finales_fusionadas = []
        st.session_state.predicciones_video_resampled = []
        st.session_state.predicciones_audio_resampled = []
        st.session_state.temp_audio_path = None


# Llama a mostrar_resultados al final del script si hay predicciones para mostrar
if st.session_state.predicciones_finales_fusionadas:
    mostrar_resultados()
else:
    st.info("ℹ️ Esperando la captura o subida de vídeo para iniciar el análisis.")
    st.markdown("---")
    st.markdown("### Estado de los Recursos:")
    # Mostrar el estado inicial de la carga de modelos para ayudar al usuario
    # El estado de carga de model_video y model_audio ya se muestra al inicio
    
    # Comprobar estado de Haar Cascade
    status_haar, success_haar = vu.init_face_cascade_classifier() # Re-chequear por si acaso
    if not success_haar:
        st.error(f"❌ Clasificador de Haar no listo: {status_haar}. La detección facial no funcionará.")
    else:
        st.success("✅ Clasificador de Haar listo.")
    
    # Comprobar estado de YAMNet y escalador (sus variables globales están en vu)
    if not vu.YAMNET_MODEL:
        st.error("❌ YAMNet no cargado. El análisis de audio puede fallar.")
    else:
        st.success("✅ YAMNet cargado.")

    if not vu.audio_scaler:
        st.error("❌ Escalador de audio no cargado. El análisis de audio puede ser impreciso.")
    else:
        st.success("✅ Escalador de audio cargado.")

    # Comprobar FFmpeg (solo una comprobación básica, la real ocurre en las funciones)
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, text=True)
        st.success("✅ FFmpeg detectado y accesible en tu PATH.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        st.error("❌ FFmpeg no detectado o no accesible en tu PATH. La extracción y guardado de audio/vídeo no funcionarán.")
    
    st.warning("☝️ Asegúrate de que todos los archivos necesarios (.h5, .npy, .xml y la carpeta 'yamnet') están en la carpeta 'model/' en el mismo directorio.")
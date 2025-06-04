import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import video_utils as vu 
import tensorflow as tf
import pandas as pd
import os
import subprocess 

# --- Configuración de la página de Streamlit ---
st.set_page_config(page_title="Lie2Me - Emociones en tiempo real", layout="wide")
st.title("🎭 Lie2Me - Detección de emociones y predisposición (Video + Audio)")

# --- Inicializar y verificar el clasificador de Haar (sin cambios) ---
face_cascade_status_message, face_cascade_loaded_successfully = vu.init_face_cascade_classifier()

if face_cascade_loaded_successfully:
    st.sidebar.success(face_cascade_status_message)
else:
    st.sidebar.error(face_cascade_status_message)
    st.stop() 

# --- Carga del modelo TensorFlow de VIDEO ---
@st.cache_resource # Caché para no cargarlo cada vez
def cargar_modelo_video():
    try:
        model_path = "model/mobilenetv2_emotion_binario_finetune.h5"
        if not os.path.exists(model_path):
            st.error(f"Error: El archivo del modelo de vídeo no se encuentra en la ruta: {model_path}")
            st.stop()
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo de vídeo de TensorFlow: {e}")
        st.stop()

model_video = cargar_modelo_video()

# --- ¡NUEVO! Carga del modelo TensorFlow de AUDIO ---
@st.cache_resource # Caché para no cargarlo cada vez
def cargar_modelo_audio():
    try:
        # ¡IMPORTANTE! Reemplaza esto con la ruta a TU modelo de audio real.
        audio_model_path = "model/audio_emotion_model.h5" 
        if not os.path.exists(audio_model_path):
            st.warning(f"Advertencia: El archivo del modelo de audio no se encuentra en la ruta: {audio_model_path}. La predicción de audio no estará disponible.")
            return None # Retorna None si no se encuentra
        model = tf.keras.models.load_model(audio_model_path)
        return model
    except Exception as e:
        st.warning(f"Error al cargar el modelo de audio de TensorFlow: {e}. La predicción de audio no estará disponible. Asegúrate de que el modelo es compatible.")
        return None

model_audio = cargar_modelo_audio()
if model_audio:
    st.sidebar.success("Modelo de audio cargado correctamente.")
else:
    st.sidebar.info("Modelo de audio no disponible. Las predicciones se realizarán solo por vídeo.")


# --- Variables para almacenar resultados globales ---
predicciones_video_totales = [] # Lista de predicciones del vídeo por cada detección facial
prediccion_audio_final = None   # Predicción única del audio de todo el vídeo
predicciones_finales_fusionadas = [] # Predicción combinada (una para vídeos subidos)
frames_procesados_para_guardar = []

# --- Sidebar: Selección de la fuente de vídeo ---
fuente = st.sidebar.radio("Selecciona la fuente de vídeo:", ("Cámara en vivo", "Subir vídeo"))

# --- ¡NUEVO! Control de ponderación en la sidebar ---
st.sidebar.markdown("---")
st.sidebar.header("Ponderación de Predicciones")
# Solo muestra los sliders si el modelo de audio está cargado
if model_audio:
    peso_video = st.sidebar.slider("Peso de la Predicción por Vídeo", 0.0, 1.0, 0.7, 0.05)
    peso_audio = st.sidebar.slider("Peso de la Predicción por Audio", 0.0, 1.0, 0.3, 0.05)
    
    # Normalizar pesos para que sumen 1 (siempre)
    total_pesos = peso_video + peso_audio
    if total_pesos > 0:
        peso_video_normalizado = peso_video / total_pesos
        peso_audio_normalizado = peso_audio / total_pesos
    else: 
        # Si ambos sliders están a 0, damos un valor por defecto para evitar división por cero
        peso_video_normalizado = 0.5
        peso_audio_normalizado = 0.5
    
    st.sidebar.info(f"Pesos normalizados: Vídeo={peso_video_normalizado:.2f}, Audio={peso_audio_normalizado:.2f}")
else:
    st.sidebar.info("La ponderación del audio está deshabilitada ya que el modelo de audio no fue cargado.")
    peso_video_normalizado = 1.0 # Si no hay audio, el vídeo tiene todo el peso
    peso_audio_normalizado = 0.0

# --- Función para procesar un solo frame (predicción de vídeo) ---
# Se le pasa el modelo de vídeo ahora
def procesar_frame(frame_input, video_model):
    boxes = vu.detectar_rostros(frame_input)
    preds_en_frame = []

    for box in boxes:
        rostro = vu.extraer_region_rostro(frame_input, box)
        if rostro is not None:
            img_proc = vu.preprocesar_imagen(rostro)
            pred = vu.predecir_emocion(video_model, img_proc) # Usamos el modelo de vídeo
            preds_en_frame.append(pred)
            
            texto = f"Predispuesto ({pred:.2f})" if pred > 0.5 else f"No predispuesto ({pred:.2f})"
            color = (0, 255, 0) if pred > 0.5 else (0, 0, 255)

            vu.dibujar_caja_y_texto(frame_input, box, texto, color=color)
        else:
            preds_en_frame.append(None)
    return frame_input, boxes, preds_en_frame

# --- Lógica principal basada en la fuente de vídeo ---
stframe = st.empty()

if fuente == "Cámara en vivo":
    st.warning("La predicción de audio no es compatible con la cámara en vivo en esta versión. Se realizará solo la predicción por vídeo.")
    st.write("🎥 Capturando desde cámara por 5 segundos...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Error: No se pudo acceder a la cámara. Asegúrate de que esté conectada y no esté en uso por otra aplicación.")
        st.stop()

    start_time = time.time()
    duration = 5
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            st.warning("No se pudo leer el frame desde la cámara. Intentando de nuevo...")
            time.sleep(0.1)
            continue

        processed_frame, current_boxes, current_preds = procesar_frame(frame, model_video)
        predicciones_video_totales.extend([p for p in current_preds if p is not None])
        frames_procesados_para_guardar.append(processed_frame.copy())

        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

    cap.release()
    # Para la cámara en vivo, la predicción combinada es simplemente la de vídeo
    predicciones_finales_fusionadas = predicciones_video_totales


elif fuente == "Subir video":
    uploaded_file = st.file_uploader("Sube un archivo de vídeo", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # 1. Guardar el archivo subido en un archivo temporal
        original_video_path = ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(uploaded_file.read())
            original_video_path = tfile.name

        # 2. Determinar la ruta para el vídeo pre-procesado (rotación)
        processed_video_path = original_video_path.replace(".mp4", "_processed.mp4")
        if not processed_video_path.endswith(".mp4"):
            processed_video_path = processed_video_path + ".mp4"
            
        st.info("Detectando y corrigiendo la orientación del vídeo (si es necesario)...")
        
        ffmpeg_command = [
            "ffmpeg",
            "-i", original_video_path,
            "-vf", "transpose=2,transpose=2,transpose=1,transpose=1", 
            "-c:v", "libx264", 
            "-preset", "fast", 
            "-crf", "23", 
            "-c:a", "aac", 
            "-b:a", "128k", 
            "-y", 
            processed_video_path
        ]
        
        try:
            subprocess.run(ffmpeg_command, capture_output=True, text=True, check=True)
            st.success("Vídeo pre-procesado para corregir orientación.")
        except subprocess.CalledProcessError as e:
            st.error(f"Error al pre-procesar el vídeo con FFmpeg: {e.stderr}")
            st.warning("Intentando procesar el vídeo original sin corrección de rotación. Podría aparecer invertido.")
            processed_video_path = original_video_path 
        except FileNotFoundError:
            st.error("Error: FFmpeg no encontrado. Asegúrate de que FFmpeg está instalado y en tu PATH.")
            st.warning("Intentando procesar el vídeo original sin corrección de rotación. Podría aparecer invertido.")
            processed_video_path = original_video_path 
        except Exception as e:
            st.error(f"Error inesperado durante el pre-procesamiento: {e}")
            st.warning("Intentando procesar el vídeo original sin corrección de rotación. Podría aparecer invertido.")
            processed_video_path = original_video_path 

        # --- ¡NUEVO! Procesamiento de Audio ---
        audio_output_path = original_video_path.replace(".mp4", ".wav")
        if model_audio: # Solo si se cargó el modelo de audio
            st.info("Extrayendo y procesando audio...")
            if vu.extract_audio_from_video(original_video_path, audio_output_path):
                audio_features = vu.preprocess_audio(audio_output_path)
                if audio_features is not None:
                    prediccion_audio_final = vu.predict_audio_emotion(model_audio, audio_features)
                    st.success(f"Predicción de Audio: {prediccion_audio_final:.2f}")
                else:
                    st.warning("No se pudieron extraer características de audio válidas.")
            else:
                st.warning("No se pudo extraer el audio del vídeo. La predicción de audio no estará disponible.")
        else:
            st.info("El modelo de audio no está cargado, se omitirá el procesamiento de audio.")

        # 3. Abrir el vídeo PROCESADO (o el original si hubo error)
        cap = cv2.VideoCapture(processed_video_path)
        
        if not cap.isOpened():
            st.error(f"Error: No se pudo abrir el archivo de vídeo: {os.path.basename(processed_video_path)}. ¿Es un archivo de vídeo válido después del procesamiento?")
            # Limpiar archivos temporales
            if os.path.exists(original_video_path): os.unlink(original_video_path)
            if os.path.exists(processed_video_path): os.unlink(processed_video_path)
            if os.path.exists(audio_output_path): os.unlink(audio_output_path) # Limpiar audio temporal
            st.stop()

        MAX_FRAME_WIDTH = 800 

        progress_bar = st.progress(0)
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0: 
            st.warning("El vídeo parece estar vacío o corrupto (0 frames) después del procesamiento.")
            cap.release()
            if os.path.exists(original_video_path): os.unlink(original_video_path)
            if os.path.exists(processed_video_path): os.unlink(processed_video_path)
            if os.path.exists(audio_output_path): os.unlink(audio_output_path)
            st.stop()

        st.write(f"Procesando vídeo: {total_frames} frames...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_height, current_width, _ = frame.shape
            if current_width > MAX_FRAME_WIDTH:
                new_width = MAX_FRAME_WIDTH
                new_height = int(current_height * (MAX_FRAME_WIDTH / current_width))
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            processed_frame, current_boxes, current_preds = procesar_frame(frame, model_video)
            predicciones_video_totales.extend([p for p in current_preds if p is not None])
            frames_procesados_para_guardar.append(processed_frame.copy())

            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB")

            frame_count += 1
            progress_bar.progress(min(100, int((frame_count / total_frames) * 100)))

        cap.release()
        # Eliminar archivos temporales al finalizar
        if os.path.exists(original_video_path): os.unlink(original_video_path)
        if os.path.exists(processed_video_path): os.unlink(processed_video_path)
        if os.path.exists(audio_output_path): os.unlink(audio_output_path) # Limpiar audio temporal

        # --- ¡NUEVO! Fusión de Predicciones ---
        if predicciones_video_totales and prediccion_audio_final is not None:
            st.markdown("---")
            st.header("Combinando Predicciones de Vídeo y Audio")
            
            # Calculamos el promedio de todas las predicciones de vídeo
            promedio_pred_video = np.mean(predicciones_video_totales)
            st.write(f"Promedio de Predicciones de Vídeo: **{promedio_pred_video:.2f}**")
            st.write(f"Predicción de Audio: **{prediccion_audio_final:.2f}**")

            # Aplicamos la fusión ponderada
            prediccion_fusionada = (promedio_pred_video * peso_video_normalizado) + \
                                   (prediccion_audio_final * peso_audio_normalizado)
            
            st.success(f"**Predicción Final Combinada (Vídeo x{peso_video_normalizado:.2f} + Audio x{peso_audio_normalizado:.2f}): {prediccion_fusionada:.2f}**")
            predicciones_finales_fusionadas.append(prediccion_fusionada) # Guardamos la predicción final

        elif predicciones_video_totales:
            st.warning("Solo se procesó la predicción de vídeo (no hay audio o el modelo de audio no se cargó).")
            # Si solo hay predicciones de vídeo, estas serán las "finales"
            predicciones_finales_fusionadas = predicciones_video_totales
        elif prediccion_audio_final is not None:
            st.warning("Solo se procesó la predicción de audio (no hay vídeo o detección de rostros).")
            # Si solo hay predicción de audio, esa será la "final"
            predicciones_finales_fusionadas.append(prediccion_audio_final)

# --- Sección de resultados y descargas (adaptada para usar las predicciones fusionadas) ---
if predicciones_finales_fusionadas:
    st.markdown("---")
    st.header("📊 Resultados del Análisis")

    # Contamos las predicciones por encima y por debajo del umbral de 0.5
    count_pos = sum(1 for p in predicciones_finales_fusionadas if p is not None and p > 0.5)
    count_neg = sum(1 for p in predicciones_finales_fusionadas if p is not None and p <= 0.5)
    
    total_preds_validas = len([p for p in predicciones_finales_fusionadas if p is not None])

    st.write(f"✅ **Detecciones Predispuestas:** {count_pos} de {total_preds_validas}")
    st.write(f"❌ **Detecciones No Predispuestas:** {count_neg} de {total_preds_validas}")

    if total_preds_validas > 0:
        # La conclusión principal se basa en el promedio o el único resultado fusionado
        if len(predicciones_finales_fusionadas) == 1 and fuente == "Subir video":
            final_pred_value = predicciones_finales_fusionadas[0]
            if final_pred_value > 0.5:
                st.success("✅ **Conclusión Final (Vídeo + Audio):** Se detecta una **predisposición a repetir la experiencia**.")
            else:
                st.error("❌ **Conclusión Final (Vídeo + Audio):** Se detecta **no predisposición a repetir la experiencia**.")
        else: # Para cámara en vivo o cuando solo hay predicciones de vídeo
            if count_pos > count_neg:
                st.success("✅ **Conclusión:** La mayoría de las detecciones indican una **predisposición a repetir la experiencia**.")
            elif count_neg > count_pos:
                st.error("❌ **Conclusión:** La mayoría de las detecciones indican **no predisposición a repetir la experiencia**.")
            else:
                st.info("ℹ️ **Conclusión:** Las detecciones de predisposición y no predisposición están equilibradas.")
    else:
        st.warning("No se pudieron obtener predicciones válidas para la conclusión.")

    if predicciones_video_totales: # Mostramos la gráfica de vídeo si hay datos
        fig_video = vu.graficar_predicciones(predicciones_video_totales, titulo="Evolución de la Predisposición por Vídeo")
        st.pyplot(fig_video)
    
    # Si hay una predicción fusionada (para vídeos subidos), la mostramos de forma destacada
    if len(predicciones_finales_fusionadas) == 1 and fuente == "Subir video":
        st.markdown("---")
        st.subheader("Predicción Final Combinada")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Predicción Final", value=f"{predicciones_finales_fusionadas[0]:.2f}")
        with col2:
            if predicciones_finales_fusionadas[0] > 0.5:
                st.success("Resultado: Predispuesto")
            else:
                st.error("Resultado: No Predispuesto")

    # CSV de predicciones (ahora incluye las columnas de audio y fusionada si aplican)
    if predicciones_video_totales:
        df_preds_dict = {
            "id_deteccion": list(range(len(predicciones_video_totales))),
            "prediccion_video_raw": predicciones_video_totales,
            "predisposicion_video": ["Predispuesto" if p > 0.5 else "No predispuesto" for p in predicciones_video_totales]
        }
        
        if prediccion_audio_final is not None:
            # Repetimos la predicción de audio para cada detección de vídeo para que el DataFrame tenga la misma longitud
            df_preds_dict["prediccion_audio_raw"] = [prediccion_audio_final] * len(predicciones_video_totales) 
            df_preds_dict["predisposicion_audio"] = ["Predispuesto" if prediccion_audio_final > 0.5 else "No predispuesto"] * len(predicciones_video_totales)
            
            # La predicción fusionada también se repite, ya que es un valor único para todo el vídeo
            df_preds_dict["prediccion_fusionada_raw"] = [predicciones_finales_fusionadas[0]] * len(predicciones_video_totales) if predicciones_finales_fusionadas else [0.0] * len(predicciones_video_totales)
            df_preds_dict["predisposicion_fusionada"] = ["Predispuesto" if p > 0.5 else "No predispuesto" for p in df_preds_dict["prediccion_fusionada_raw"]]

        df_preds = pd.DataFrame(df_preds_dict)
        csv = df_preds.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Descargar predicciones CSV",
            data=csv,
            file_name="predicciones_lie2me.csv",
            mime="text/csv"
        )

    # ... (El resto del código para guardar el vídeo anotado permanece igual) ...

else:
    st.info("Esperando la captura o subida de vídeo para iniciar el análisis. No se han procesado frames aún.")
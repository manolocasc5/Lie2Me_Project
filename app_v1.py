import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import video_utils as vu # Importa tu módulo de utilidades
import tensorflow as tf
import pandas as pd
import os
import subprocess # Necesario para ejecutar comandos FFmpeg

# --- Configuración de la página de Streamlit ---
st.set_page_config(page_title="Lie2Me - Emociones en tiempo real", layout="wide")
st.title("🎭 Lie2Me - Detección de emociones y predisposición")

# --- Inicializar y verificar el clasificador de Haar ---
face_cascade_status_message, face_cascade_loaded_successfully = vu.init_face_cascade_classifier()

if face_cascade_loaded_successfully:
    st.sidebar.success(face_cascade_status_message)
else:
    st.sidebar.error(face_cascade_status_message)
    st.stop() # Detener la aplicación si el clasificador no carga, ya que la detección facial es crítica

# --- Carga del modelo TensorFlow (cacheado para eficiencia) ---
@st.cache_resource
def cargar_modelo():
    try:
        model_path = "model/mobilenetv2_emotion_binario_finetune.h5"
        if not os.path.exists(model_path):
            st.error(f"Error: El archivo del modelo no se encuentra en la ruta: {model_path}")
            st.stop()
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo de TensorFlow: {e}")
        st.info("Asegúrate de que 'model/mobilenetv2_emotion_binario_finetune.h5' existe y es compatible con tu versión de TensorFlow.")
        st.stop()

model = cargar_modelo()

# --- Variables para almacenar resultados globales ---
predicciones_totales = []
frames_procesados_para_guardar = []

# --- Sidebar: Selección de la fuente de video ---
fuente = st.sidebar.radio("Selecciona la fuente de video:", ("Cámara en vivo", "Subir video"))

# --- Función para procesar un solo frame ---
def procesar_frame(frame_input):
    boxes = vu.detectar_rostros(frame_input)
    preds_en_frame = []

    for box in boxes:
        rostro = vu.extraer_region_rostro(frame_input, box)
        if rostro is not None:
            img_proc = vu.preprocesar_imagen(rostro)
            pred = vu.predecir_emocion(model, img_proc)
            preds_en_frame.append(pred)
            
            texto = f"Predispuesto ({pred:.2f})" if pred > 0.5 else f"No predispuesto ({pred:.2f})"
            color = (0, 255, 0) if pred > 0.5 else (0, 0, 255)

            vu.dibujar_caja_y_texto(frame_input, box, texto, color=color)
        else:
            preds_en_frame.append(None)
    return frame_input, boxes, preds_en_frame

# --- Lógica principal basada en la fuente de video ---
stframe = st.empty()

if fuente == "Cámara en vivo":
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

        processed_frame, current_boxes, current_preds = procesar_frame(frame)
        predicciones_totales.extend([p for p in current_preds if p is not None])
        frames_procesados_para_guardar.append(processed_frame.copy())

        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

    cap.release()

elif fuente == "Subir video":
    uploaded_file = st.file_uploader("Sube un archivo de video", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # 1. Guardar el archivo subido en un archivo temporal
        original_video_path = ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(uploaded_file.read())
            original_video_path = tfile.name

        # 2. Determinar la ruta para el video procesado
        processed_video_path = original_video_path.replace(".mp4", "_processed.mp4")
        # Asegúrate de que la extensión sea .mp4 para la salida, aunque la entrada sea .MOV
        if not processed_video_path.endswith(".mp4"):
            processed_video_path = processed_video_path + ".mp4"
            
        st.info("Detectando y corrigiendo la orientación del video (si es necesario)...")
        
        # --- APLICAR TRANSFORMACIÓN DE ROTACIÓN USANDO FFmpeg ---
        # Aquí es donde integramos la lógica de corrección que encontraste.
        # Basado en tu experiencia, usaremos el comando que te funcionó.
        # Es crucial que FFmpeg esté accesible en el PATH del sistema donde se ejecuta la app.
        
        # Puedes intentar el comando que te funcionó:
        ffmpeg_command = [
            "ffmpeg",
            "-i", original_video_path,
            "-vf", "transpose=2,transpose=2,transpose=1,transpose=1", # El filtro que te funcionó
            "-c:v", "libx264", # Recodificar video (libx264 es un códec MP4 común)
            "-preset", "fast", # Velocidad de codificación (puedes ajustar)
            "-crf", "23", # Calidad (menor valor = mayor calidad/tamaño)
            "-c:a", "aac", # Recodificar audio a AAC para compatibilidad MP4
            "-b:a", "128k", # Bitrate de audio
            "-y", # Sobrescribir archivo de salida si existe
            processed_video_path
        ]
        
        # O, si sabes que el problema es la displaymatrix y quieres ignorarla:
        # ffmpeg_command = [
        #     "ffmpeg",
        #     "-noautorotate", # Ignorar metadatos de rotación
        #     "-i", original_video_path,
        #     "-vf", "transpose=1", # O el transpose que sea necesario para este caso
        #     "-c:v", "libx264", 
        #     "-preset", "fast", 
        #     "-crf", "23", 
        #     "-c:a", "aac", 
        #     "-b:a", "128k", 
        #     "-y", 
        #     processed_video_path
        # ]

        try:
            # Ejecutar el comando FFmpeg
            process = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=True)
            st.success("Video pre-procesado para corregir orientación.")
            # Opcional: mostrar stdout/stderr de ffmpeg para depuración
            # st.text("FFmpeg STDOUT:\n" + process.stdout)
            # st.text("FFmpeg STDERR:\n" + process.stderr)

        except subprocess.CalledProcessError as e:
            st.error(f"Error al pre-procesar el video con FFmpeg: {e.stderr}")
            st.warning("Intentando procesar el video original sin corrección de rotación. Podría aparecer invertido.")
            processed_video_path = original_video_path # Volver al original si falla la corrección
        except FileNotFoundError:
            st.error("Error: FFmpeg no encontrado. Asegúrate de que FFmpeg está instalado y en tu PATH.")
            st.warning("Intentando procesar el video original sin corrección de rotación. Podría aparecer invertido.")
            processed_video_path = original_video_path # Volver al original si FFmpeg no está
        except Exception as e:
            st.error(f"Error inesperado durante el pre-procesamiento: {e}")
            st.warning("Intentando procesar el video original sin corrección de rotación. Podría aparecer invertido.")
            processed_video_path = original_video_path # Volver al original si hay otro error

        # 3. Abrir el video PROCESADO (o el original si hubo error)
        cap = cv2.VideoCapture(processed_video_path)
        
        if not cap.isOpened():
            st.error(f"Error: No se pudo abrir el archivo de video: {os.path.basename(processed_video_path)}. ¿Es un archivo de video válido después del procesamiento?")
            # Limpia archivos temporales
            if os.path.exists(original_video_path): os.unlink(original_video_path)
            if os.path.exists(processed_video_path): os.unlink(processed_video_path)
            st.stop()

        # Ya no necesitamos get_video_rotation aquí porque FFmpeg ya fijo la rotación
        rotation_angle = 0 # Asumimos 0 porque ya está rotado a nivel de píxel
        
        MAX_FRAME_WIDTH = 800 

        progress_bar = st.progress(0)
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0: 
            st.warning("El video parece estar vacío o corrupto (0 frames) después del procesamiento.")
            cap.release()
            if os.path.exists(original_video_path): os.unlink(original_video_path)
            if os.path.exists(processed_video_path): os.unlink(processed_video_path)
            st.stop()

        st.write(f"Procesando video: {total_frames} frames...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # No es necesario aplicar vu.rotate_frame_if_needed aquí
            # porque FFmpeg ya lo hizo a nivel de píxel.
            
            current_height, current_width, _ = frame.shape
            if current_width > MAX_FRAME_WIDTH:
                new_width = MAX_FRAME_WIDTH
                new_height = int(current_height * (MAX_FRAME_WIDTH / current_width))
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            processed_frame, current_boxes, current_preds = procesar_frame(frame)
            predicciones_totales.extend([p for p in current_preds if p is not None])
            frames_procesados_para_guardar.append(processed_frame.copy())

            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB")

            frame_count += 1
            progress_bar.progress(min(100, int((frame_count / total_frames) * 100)))

        cap.release()
        # Eliminar ambos archivos temporales al finalizar
        if os.path.exists(original_video_path): os.unlink(original_video_path)
        if os.path.exists(processed_video_path): os.unlink(processed_video_path)

# --- Resto de la sección de resultados y descargas (sin cambios funcionales importantes) ---
if predicciones_totales:
    st.markdown("---")
    st.header("📊 Resultados del Análisis")

    count_pos = sum(1 for p in predicciones_totales if p is not None and p > 0.5)
    count_neg = sum(1 for p in predicciones_totales if p is not None and p <= 0.5)
    
    total_preds_validas = len([p for p in predicciones_totales if p is not None])

    st.write(f"✅ **Frames/Rostros Predispuestos:** {count_pos} de {total_preds_validas}")
    st.write(f"❌ **Frames/Rostros No Predispuestos:** {count_neg} de {total_preds_validas}")

    if total_preds_validas > 0:
        if count_pos > count_neg:
            st.success("✅ **Conclusión:** La mayoría de las detecciones indican una **predisposición a repetir la experiencia**.")
        elif count_neg > count_pos:
            st.error("❌ **Conclusión:** La mayoría de las detecciones indican **no predisposición a repetir la experiencia**.")
        else:
            st.info("ℹ️ **Conclusión:** Las detecciones de predisposición y no predisposición están equilibradas.")
    else:
        st.warning("No se pudieron obtener predicciones válidas para la conclusión.")

    if predicciones_totales:
        fig = vu.graficar_predicciones(predicciones_totales, titulo="Evolución de la Predisposición por Detección")
        st.pyplot(fig)

    if predicciones_totales:
        df_preds = pd.DataFrame({
            "id_deteccion": list(range(len(predicciones_totales))),
            "prediccion_raw": predicciones_totales,
            "predisposicion": ["Predispuesto" if p > 0.5 else "No predispuesto" for p in predicciones_totales]
        })
        csv = df_preds.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Descargar predicciones CSV",
            data=csv,
            file_name="predicciones_lie2me.csv",
            mime="text/csv"
        )

    if frames_procesados_para_guardar:
        output_video_filename = "video_lie2me_anotado.mp4"
        try:
            # Los FPS deben ser los del video original o un valor por defecto si no se pudo leer.
            # Aquí, 'cap' es el del video *procesado*. Asumimos que su FPS es el correcto.
            fps = cap.get(cv2.CAP_PROP_FPS) if 'cap' in locals() and cap.isOpened() else 15
            if fps == 0: 
                fps = 15 
            
            success = vu.guardar_video(frames_procesados_para_guardar, output_video_filename, fps=fps)
            if success:
                with open(output_video_filename, "rb") as f:
                    st.download_button(
                        label="⬇️ Descargar video anotado",
                        data=f,
                        file_name=output_video_filename,
                        mime="video/mp4"
                    )
                os.remove(output_video_filename)
            else:
                st.error("No se pudo generar el video anotado para descargar.")
        except Exception as e:
            st.error(f"Error al intentar guardar o descargar el video anotado: {e}")
            st.info("Asegúrate de que no haya un video con el mismo nombre abierto o de que haya suficiente espacio en disco.")
else:
    st.info("Esperando la captura o subida de video para iniciar el análisis. No se han procesado frames aún.")
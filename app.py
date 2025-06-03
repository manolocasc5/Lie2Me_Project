import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import video_utils as vu
import tensorflow as tf
from PIL import Image
import pandas as pd
import os

st.set_page_config(page_title="Lie2Me - Emociones en tiempo real", layout="wide")
st.title("🎭 Lie2Me - Detección de emociones y predisposición")

# Cargar modelo entrenado
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("model/mobilenetv2_emotion_binario_finetune.h5")

model = cargar_modelo()

# Sidebar: elegir fuente de video
fuente = st.sidebar.radio("Selecciona la fuente de video:", ("Cámara en vivo", "Subir video"))

# Variables para almacenar resultados
predicciones_totales = []
boxes_totales = []
frames_procesados = []


def procesar_frame(frame):
    """Detecta rostros, predice emociones y retorna frame anotado + predicciones."""
    boxes = vu.detectar_rostros(frame)
    preds = []

    for box in boxes:
        rostro = vu.extraer_region_rostro(frame, box)
        if rostro is not None:
            img_proc = vu.preprocesar_imagen(rostro)
            pred = vu.predecir_emocion(model, img_proc)
            preds.append(pred)
            texto = "Predispuesto" if pred > 0.5 else "No predispuesto"
            color = (0, 255, 0) if pred > 0.5 else (0, 0, 255)
            vu.dibujar_caja_y_texto(frame, box, texto, color=color)
        else:
            preds.append(None)
    return frame, boxes, preds

# Función para guardar video
def guardar_video(frames, path, fps=10):
    if not frames:
        return
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for f in frames:
        out.write(f)
    out.release()

if fuente == "Cámara en vivo":
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    duration = 5  # segundos

    st.write("🎥 Capturando desde cámara por 5 segundos...")

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            st.error("No se pudo leer frame desde la cámara.")
            break

        frame, boxes, preds = procesar_frame(frame)
        predicciones_totales.extend([p for p in preds if p is not None])
        boxes_totales.append(boxes)
        frames_procesados.append(frame.copy())

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

    cap.release()

elif fuente == "Subir video":
    uploaded_file = st.file_uploader("Sube un archivo de video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, boxes, preds = procesar_frame(frame)
            predicciones_totales.extend([p for p in preds if p is not None])
            boxes_totales.append(boxes)
            frames_procesados.append(frame.copy())

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB")

        cap.release()

if predicciones_totales:
    # Estadísticas y gráfico
    count_pos = sum(p > 0.5 for p in predicciones_totales)
    count_neg = len(predicciones_totales) - count_pos

    st.write(f"✅ Frames predispuestos: {count_pos}")
    st.write(f"❌ Frames no predispuestos: {count_neg}")

    if count_pos > count_neg:
        st.success("✅ Predicción final: Predispuesto a repetir experiencia")
    else:
        st.error("❌ Predicción final: No predispuesto a repetir experiencia")

    fig = vu.graficar_predicciones(predicciones_totales, titulo="Evolución de la Predisposición")
    st.pyplot(fig)

    # Descargar CSV con predicciones por frame
    df = pd.DataFrame({"frame": list(range(len(predicciones_totales))), "prediccion": predicciones_totales})
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar predicciones CSV", data=csv, file_name="predicciones.csv", mime="text/csv")

    # Descargar video procesado
    video_path = "video_procesado.mp4"
    guardar_video(frames_procesados, video_path)
    with open(video_path, "rb") as f:
        st.download_button("⬇️ Descargar video anotado", data=f, file_name="video_procesado.mp4", mime="video/mp4")
else:
    st.info("No se procesaron frames aún.")

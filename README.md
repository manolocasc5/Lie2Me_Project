# Lie2Me - Detección de emociones y predisposición a repetir experiencia

Este proyecto es una aplicación de análisis facial para detectar emociones en clientes y predecir si están predispuestos a repetir una experiencia (por ejemplo, en la industria hotelera). Utiliza transferencia de aprendizaje con MobileNetV2 fine-tuneado para clasificación binaria basada en emociones faciales.

---

## Características principales

- Detección de rostros en video o cámara en vivo usando `face_recognition`.
- Clasificación binaria de predisposición basada en emociones faciales.
- Interfaz interactiva con Streamlit.
- Soporta entrada de video desde archivo o cámara en tiempo real.
- Visualización de cajas con etiquetas de emoción sobre los rostros detectados.
- Gráfico de evolución de emociones durante la sesión.
- Exportación de predicciones en CSV.
- Guardado opcional de video con anotaciones.

---

## Estructura del proyecto

├── app.py # App principal de Streamlit

├── video_utils.py # Funciones para detección, preprocesado y visualización

├── model/ # Carpeta con modelo entrenado .h5

├── requirements.txt # Dependencias del proyecto

└── README.md # Documentación


---

## Cómo usar

### Entrenamiento

Se asume que el modelo `mobilenetv2_emotion_binario_finetune.h5` ya está entrenado y guardado en la carpeta `model/`.

### Ejecutar la app

1. Instala dependencias:

```bash
pip install -r requirements.txt
```

2. Ejecuta la app Streamlit:

```bash
streamlit run app.py
```

3. En la interfaz web puedes:

- Elegir "Cámara en vivo" para capturar video desde la webcam.

- Subir un archivo de video para analizarlo.

- Visualizar predicciones en tiempo real y descarga de resultados.

### Tecnologías
- TensorFlow/Keras (Transfer Learning con MobileNetV2)

- OpenCV (procesamiento de video)

- face_recognition (detección facial)

- Streamlit (interfaz web interactiva)

- Matplotlib (visualización de gráficos)

### Requisitos
- Python 3.7+

- GPU recomendada para entrenamiento (opcional para inferencia)

## Licencia
MIT License

## Autores
Tu Nombre - [Tu Email o GitHub]

---

## requirements.txt

- tensorflow>=2.11.0
- opencv-python-headless
- face_recognition
- streamlit
- matplotlib
- numpy
- pandas
- Pillow


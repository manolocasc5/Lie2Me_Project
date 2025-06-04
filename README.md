# Lie2Me - Detección de Predisposición (Vídeo y Audio)
Este proyecto es una aplicación de análisis multimodal que combina el procesamiento de vídeo y audio para detectar emociones y predecir la predisposición de un individuo (por ejemplo, en el contexto de la interacción con clientes). Utiliza modelos de aprendizaje profundo para el análisis facial y de voz, ofreciendo una visión más completa que el análisis de un solo componente.

## 🌟 Características Principales
- Análisis Multimodal: Combina la detección de emociones faciales (vídeo) y la predisposición emocional basada en el tono de voz (audio) para una predicción fusionada.

- Detección Facial Robusta: Utiliza el clasificador Haar Cascade de OpenCV para la detección de rostros en tiempo real.

- Modelos de Transferencia de Aprendizaje:

    - Vídeo: Modelo MobileNetV2 fine-tuneado para clasificación binaria de predisposición basada en expresiones faciales.

    - Audio: Modelo de red neuronal densa entrenado sobre embeddings de YAMNet (modelo pre-entrenado de Google) para clasificación binaria de predisposición vocal.

- Extracción y Procesamiento de Audio: Utiliza FFmpeg para extraer la pista de audio de los vídeos subidos y librosa para su preprocesamiento y segmentación.

- Interfaz Interactiva con Streamlit: Aplicación web amigable para la interacción con el usuario.

- Soporte de Entrada Flexible:

    - Cámara en Vivo: Captura de vídeo desde la webcam con límite de tiempo (ej. 10 segundos) y redimensionamiento para un mejor rendimiento.

    - Subida de Vídeo: Análisis de vídeos desde archivos locales (.mp4, .avi, .mov).

- Visualización Detallada:

    - Gráficos de evolución temporal para las predicciones de vídeo, audio y la fusión de ambos.

    - Representación de cajas delimitadoras con etiquetas de predisposición sobre los rostros detectados en el vídeo.

- Exportación de Resultados: Opción para guardar el vídeo procesado con las anotaciones faciales.

- Control de Ponderación: Ajusta el peso de las predicciones de vídeo y audio en la fusión final.

## 📁 Estructura del Proyecto
.
├── app.py                     # Aplicación principal de Streamlit

├── video_utils.py             # Funciones auxiliares para procesamiento de vídeo y audio

├── requirements.txt           # Lista de dependencias de Python

├── README.md                  # Este archivo de documentación

└── model/                     # Carpeta para modelos entrenados y archivos auxiliares

    ├── mobilenetv2_emotion_binario_finetune.h5 # Modelo de vídeo (predicción facial)
    
    ├── audio_emotion_model.h5 # Modelo de audio (predicción vocal)
    
    ├── audio_scaler.npy       # Escalador para los embeddings de audio
    
    ├── haarcascade_frontalface_default.xml # Clasificador de Haar para detección facial
    
    └── yamnet/                # Carpeta que contiene el modelo YAMNet de TensorFlow Hub
    
        ├── saved_model.pb
        
        └── variables/
        
        └── assets/
        
        └── ... (otros archivos de YAMNet)

## 🚀 Cómo Usar
### 🛠️ Entrenamiento (Preparación de Modelos)
Se asume que los modelos mobilenetv2_emotion_binario_finetune.h5 y audio_emotion_model.h5 ya están entrenados y guardados en la carpeta model/. Además, el escalador audio_scaler.npy debe haber sido generado y guardado en la misma carpeta.

Nota: El modelo YAMNet (yamnet/) debe descargarse de TensorFlow Hub y colocarse en la carpeta model/.

### 🏃 Ejecutar la Aplicación
#### 1- Clona el repositorio:

git clone <URL_DE_TU_REPOSITORIO>
cd <nombre_del_proyecto>

#### 2- Crea y activa un entorno virtual (muy recomendado):

python -m venv venv

##### En Linux/macOS:
source venv/bin/activate
##### En Windows:
venv\Scripts\activate

#### 3- Instala las dependencias de Python:

pip install -r requirements.txt

#### 4- Instala FFmpeg:
FFmpeg es una herramienta externa esencial para la extracción de audio de vídeos y para guardar los vídeos procesados. No se instala con pip.

- En Linux (Ubuntu/Debian):

sudo apt update
sudo apt install ffmpeg

- En macOS (con Homebrew):

brew install ffmpeg

- En Windows:
Descarga los binarios precompilados desde ffmpeg.org/download.html, descomprímelos y añade la ruta a la carpeta bin de FFmpeg a las variables de entorno PATH de tu sistema. (Busca tutoriales específicos para "añadir FFmpeg al PATH en Windows" si necesitas ayuda).

#### 5- Asegúrate de que los modelos están en la carpeta model/:
Verifica que todos los archivos .h5, .npy, .xml y la carpeta yamnet/ estén correctamente ubicados dentro del directorio model/.

#### 6- Ejecuta la aplicación Streamlit:

streamlit run app.py

#### 7- En la Interfaz Web (tu navegador):

    - Por defecto, la opción "Subir vídeo" estará seleccionada. Puedes subir un archivo de vídeo (.mp4, .avi, .mov) para un análisis completo (vídeo + audio).

    - Alternativamente, puedes elegir "Cámara en vivo" en la barra lateral para capturar vídeo desde tu webcam. La captura se detendrá automáticamente después de un tiempo predefinido para el análisis de vídeo.

    - Ajusta los pesos de las predicciones de vídeo y audio en la barra lateral para influir en la fusión final.

    - Visualiza los resultados detallados en gráficos y descarga el vídeo procesado con las anotaciones.

## 💻 Tecnologías Utilizadas
- Python 3.9+

- TensorFlow / Keras: Para la construcción y ejecución de modelos de aprendizaje profundo (MobileNetV2, Red Neuronal Densa para audio).

- TensorFlow Hub: Para cargar el modelo YAMNet pre-entrenado.

- OpenCV (opencv-python): Para procesamiento de vídeo, detección facial (Haar Cascade) y dibujo de anotaciones.

- Streamlit: Para la creación de la interfaz de usuario web interactiva.

- FFmpeg: Herramienta externa de línea de comandos para la extracción de audio de vídeo y el guardado de vídeo procesado.

- Librosa: Para el preprocesamiento y análisis de señales de audio.

- SoundFile: Soporte para lectura/escritura de archivos de audio (usado por Librosa).

- NumPy: Computación numérica eficiente.

- Pandas: Manipulación y análisis de datos.

- Matplotlib: Generación de gráficos y visualizaciones.

- Scikit-learn: Para el escalado de características (StandardScaler).

- Joblib: Para guardar y cargar objetos Python eficientemente (el escalador).

## 📦 requirements.txt
### Core application framework
streamlit==1.30.0

### Deep Learning Framework
tensorflow==2.15.0
tensorflow-hub==0.16.1

### Video Processing
opencv-python==4.9.0.80

### Audio Processing
librosa==0.10.1
soundfile==0.12.1

### Numerical Computing & Data Handling
numpy==1.26.4
pandas==2.2.0
matplotlib==3.8.3
scikit-learn==1.4.0
joblib==1.3.2

## 📄 Licencia
MIT License

## ✒️ Autores
Alejandro Cabrera y Manolo Castillo

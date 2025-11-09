# Plataforma de Autenticación Biométrica con Video

## Descripción

Plataforma de autenticación biométrica de última generación que utiliza **video único** para capturar características faciales y de voz simultáneamente. Los usuarios se registran y autentican mediante un video de 10-15 segundos, eliminando la necesidad de archivos separados de imagen y audio.

## 🎥 Características Principales

- ✅ **Registro con Video Único**: Un solo video captura cara y voz
- ✅ **Autenticación Basada en Username**: Sistema simplificado sin contraseñas
- ✅ **Entrenamiento Automático**: Modelos se actualizan automáticamente tras cada registro
- ✅ **Procesamiento Asíncrono**: Entrenamiento en segundo plano, sin bloqueo
- ✅ **Doble Validación Biométrica**: Facial + Voz para mayor seguridad
- ✅ **Interfaz Moderna**: Grabación de video integrada en el navegador

## Tecnologías Utilizadas

### Backend
- Django 5.0+
- PostgreSQL
- OpenCV (procesamiento de video)
- FFmpeg (extracción de audio)

### Machine Learning
- FaceNet (PyTorch) - Reconocimiento facial
- SpeechBrain - Reconocimiento de voz
- Scikit-learn (SVM) - Clasificación
- Torch & Torchaudio
- Hugging Face Hub

### Frontend
- MediaRecorder API (grabación de video)
- WebM/MP4 video support
- Vanilla JavaScript

## 📋 Requisitos del Sistema

- Python 3.12
- PostgreSQL
- FFmpeg (instalación del sistema)
- Navegador moderno (Chrome, Firefox, Edge, Safari)

## 🚀 Instalación

### 1. Clonar el Repositorio
```bash
git clone <repository-url>
cd facial-recognition
```

### 2. Instalar FFmpeg
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Descargar desde https://ffmpeg.org/download.html
```

### 3. Instalar Dependencias de Python
```bash
uv sync
```

### 4. Configurar Base de Datos
```bash
# Crear base de datos PostgreSQL
createdb facial_recognition

# Ejecutar migraciones
cd src
python manage.py migrate
```

### 5. Entrenar Modelos Iniciales (Opcional)
Si tienes datos de entrenamiento existentes:
```bash
make train-facial
make train-voice
```

### 6. Iniciar Servidor
```bash
python src/manage.py runserver
```

Accede a: `http://localhost:8000`

## 💡 Uso

### Registro de Usuario

1. Navega a `/register`
2. Ingresa un **username único**
3. Haz clic en "Start Video Recording"
4. Durante la grabación (10-15 segundos):
   - Mueve tu cabeza suavemente (izquierda/derecha, arriba/abajo)
   - Di claramente:
     - "Soy [tu nombre]"
     - "Esta es mi voz"
     - "Estoy registrándome en el sistema"
5. Revisa el video y haz clic en "Complete Registration"
6. El sistema procesará el video y entrenará los modelos automáticamente

### Autenticación

1. Navega a `/login`
2. Ingresa tu **username**
3. Graba un nuevo video (10-15 segundos) con las mismas características
4. El sistema validará tus datos biométricos
5. Acceso concedido si la validación es exitosa

## 🏗️ Arquitectura

```
┌─────────────────┐
│   Frontend      │
│  (Video Input)  │
└────────┬────────┘
         │ video file
         ▼
┌─────────────────┐
│   Django Views  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  VideoProcessor         │
│  - Extract frames (2fps)│
│  - Extract audio        │
│  - Segment audio clips  │
└────────┬────────────────┘
         │
         ├─► models/facial/data/{username}/*.png
         └─► models/voice/data/{username}/*.wav
         │
         ▼
┌─────────────────────────┐
│  TrainingService        │
│  (Async Background)     │
│  - Train facial model   │
│  - Train voice model    │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Model Checkpoints      │
│  - SVM classifiers      │
│  - Label encoders       │
└─────────────────────────┘
```

## 📁 Estructura del Proyecto

```
facial-recognition/
├── models/
│   ├── facial/
│   │   ├── data/           # Frames extraídos por usuario
│   │   ├── checkpoints/    # Modelos entrenados
│   │   └── pipeline.py     # Script de entrenamiento
│   └── voice/
│       ├── data/           # Clips de audio por usuario
│       ├── checkpoints/    # Modelos entrenados
│       └── pipeline.py     # Script de entrenamiento
├── src/
│   └── authentication/
│       ├── services/
│       │   ├── video_processor.py     # Procesamiento de video
│       │   ├── training_service.py    # Entrenamiento asíncrono
│       │   ├── registration_service.py
│       │   └── authentication_service.py
│       ├── templates/
│       │   └── authentication/
│       │       ├── register.html      # Registro con video
│       │       └── login.html         # Login con video
│       └── models.py                   # User model actualizado
├── pyproject.toml
└── VIDEO_AUTHENTICATION_GUIDE.md       # Documentación detallada
```

## 🔧 Configuración

### Parámetros de VideoProcessor
Ubicación: `src/authentication/services/video_processor.py`

```python
FRAMES_PER_SECOND = 2          # Tasa de extracción de frames
MIN_FRAMES = 10                 # Mínimo de frames requeridos
AUDIO_SEGMENT_DURATION = 3      # Duración de clips de audio
```

### Restricciones de Grabación
Ubicación: Templates de frontend

```javascript
MIN_RECORDING_DURATION = 10000  // 10 segundos
MAX_RECORDING_DURATION = 15000  // 15 segundos
```

## 📊 Rendimiento

| Operación             | Tiempo Estimado |
| --------------------- | --------------- |
| Subida de video       | 2-5 segundos    |
| Extracción de frames  | 1-2 segundos    |
| Extracción de audio   | 1-2 segundos    |
| Validación biométrica | 2-3 segundos    |
| Entrenamiento (async) | 30-120 segundos |

## 🔒 Seguridad

- Sin almacenamiento de contraseñas (100% biométrico)
- Videos temporales eliminados tras procesamiento
- Transmisión HTTPS obligatoria en producción
- Modelos almacenados de forma segura en servidor
- Gestión de sesiones Django estándar

## 🛠️ Entrenamiento de Modelos

### Entrenamiento Manual
```bash
make train-facial
make train-voice
```

### Entrenamiento Automático
- Se activa automáticamente tras cada registro
- Se ejecuta en segundo plano (no bloquea)
- Thread-safe con mecanismo de bloqueo
- Logs disponibles en servicios

## 📖 Documentación Adicional

Para guía detallada, ver: [VIDEO_AUTHENTICATION_GUIDE.md](./VIDEO_AUTHENTICATION_GUIDE.md)

Incluye:
- Flujo completo de registro/autenticación
- Arquitectura técnica detallada
- Troubleshooting
- Configuración avanzada
- Mejoras futuras

## 🐛 Solución de Problemas

### "Failed to open video file"
- Verifica instalación de FFmpeg: `ffmpeg -version`
- Comprueba formato de video compatible

### "Biometric validation failed"
- Asegura buena iluminación
- Habla claramente durante grabación
- Mantén cara visible todo el tiempo

### Entrenamiento lento
- Primera vez con muchos usuarios toma más tiempo
- Entrenamientos subsecuentes son incrementales
- Verifica proceso Python no está colgado

## 🤝 Contribución

1. Fork del proyecto
2. Crea feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a branch (`git push origin feature/AmazingFeature`)
5. Abre Pull Request

## 📝 Licencia

[Especificar licencia]

## ✨ Mejoras Futuras

- [ ] Detección de vida (anti-replay)
- [ ] Autenticación multi-factor opcional
- [ ] Preprocesamiento de calidad de video
- [ ] Feedback en tiempo real de calidad de frame
- [ ] Actualizaciones progresivas de modelos
- [ ] Soporte para múltiples dispositivos por usuario

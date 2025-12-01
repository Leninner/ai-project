# Multimodal Biometric Authentication System

This project is a multimodal biometric authentication system that uses a single video to capture both facial and voice characteristics for user authentication. The system is built with Python, Django, and a number of machine learning and deep learning libraries.

## Features

- **Single Video Registration**: Captures both facial and voice data from a single 10-15 second video.
- **Passwordless Authentication**: Users are authenticated based on their username and biometric data.
- **Automatic Training**: The models are automatically updated after each new user registration.
- **Asynchronous Processing**: Model training is done in the background to avoid blocking the application.
- **Dual Biometric Validation**: Uses both facial and voice recognition for enhanced security.
- **Modular Architecture**: Allows for interchangeable classifiers (e.g., Neural Networks, CNNs).
- **Liveness Detection**: The system can detect liveness to prevent spoofing attacks.

## System Architecture

The system is composed of a web application built with Django that handles user registration and authentication. When a user registers, a video is recorded and sent to the server. The server then processes the video to extract facial frames and audio segments. These are used to train the facial and voice recognition models.

During authentication, the user records a new video, which is then compared against the trained models to verify the user's identity.

### General Architecture

```mermaid
graph TB
    A[Cliente Web] -->|Video WebM/MP4| B[Django Views]
    B --> C[VideoProcessor]
    C -->|Extracción| D[Frames PNG]
    C -->|Extracción| E[Audio WAV]
    D --> F[Facial Embedding Extractor/Raw Image]
    E --> G[Voice Embedding Extractor]
    F -->|FaceNet 512-dim / Raw Image| H[Facial Classifier]
    G -->|SpeechBrain 192-dim| I[Voice Classifier]
    H -->|Neural Network/CNN| J[Biometric Validator]
    I -->|SVM/NN| J
    J -->|Resultado| K[Autenticación]
    
    C -->|Registro| L[Training Service]
    L -->|Async| M[Pipeline Training]
    M --> N[Model Checkpoints]
    N --> H
    N --> I
```

### Video Processing Pipeline

```mermaid
sequenceDiagram
    participant U as Usuario
    participant D as Django
    participant VP as VideoProcessor
    participant FE as Facial Extractor
    participant VE as Voice Extractor
    participant BC as Biometric Classifier
    participant TS as Training Service
    
    U->>D: Sube video (registro/login)
    D->>VP: process_video()
    VP->>VP: Conversión WebM→MP4 (FFmpeg)
    VP->>VP: Extracción frames (100 frames)
    VP->>VP: Detección rostro (MTCNN)
    VP->>VP: Extracción audio (FFmpeg)
    VP->>VP: Segmentación audio (3s clips)
    
    alt Registro
        VP->>VP: Guardar frames en data/{username}/
        VP->>VP: Guardar audio en data/{username}/
        VP->>TS: trigger_async_training()
        TS->>TS: Entrenar modelos (background)
    else Autenticación
        VP->>FE: extract_embedding(frame) / identify_face_from_image(frame)
        VP->>VE: extract_embedding(audio)
        FE->>BC: identify_face(embedding/image)
        VE->>BC: identify_speaker(embedding)
        BC->>D: Resultado validación
    end
```

### Classifier Architecture

```mermaid
graph LR
    A[Embedding Vector/Raw Image] --> B{Classifier Type}
    B -->|Neural Network| C[Neural Network]
    B -->|CNN| D[CNN Classifier]
    C --> E[Label Encoder]
    D --> F[Label Encoder]
    E --> G[Prediction + Confidence]
    F --> G
    G --> H{Confidence > Threshold?}
    H -->|Yes| I[Identified User]
    H -->|No| J[Unknown]
```

## System Components

### Video Processing

The `VideoProcessor` component is responsible for extracting biometric data from the input video.

#### Frame Extraction

- **Technology**: OpenCV + MTCNN (FaceNet)
- **Parameters**:
  - `REQUIRED_FRAMES = 100`: Number of frames with a detected face required.
  - `MIN_VIDEO_DURATION_SECONDS = 30`: Minimum video duration.
  - `image_size = 160`: Image size for FaceNet.
  - `min_face_size = 20`: Minimum detected face size.

**Process**:
1. Convert WebM to MP4 (if necessary) using FFmpeg.
2. Read the video frame by frame.
3. Detect a face in each frame using MTCNN.
4. Crop and resize the face.
5. Save as a PNG image (160x160 pixels).

#### Audio Extraction

- **Technology**: FFmpeg + SoundFile
- **Parameters**:
  - `AUDIO_SEGMENT_DURATION = 3`: Duration of each audio segment in seconds.
  - `MIN_AUDIO_SEGMENTS = 10`: Minimum required audio segments.
  - `MAX_AUDIO_SEGMENTS = 15`: Maximum allowed audio segments.
  - `sample_rate = 16000`: Sample rate for SpeechBrain.

**Process**:
1. Extract the audio track from the video using FFmpeg.
2. Convert to WAV format (16-bit PCM, mono, 16kHz).
3. Segment the audio into 3-second clips.
4. Save each segment as an individual WAV file.

### Embedding and Image Processing

#### Facial Embeddings (FaceNet) - For SVM/NN Classifiers

**Model**: InceptionResnetV1 pre-trained on VGGFace2

- **Architecture**: Inception-ResNet convolutional neural network.
- **Embedding Dimension**: 512 features.
- **Preprocessing**:
  - Face detection with MTCNN.
  - Normalization to 160x160 pixels.
  - Pixel value normalization.

#### Facial Image Processing - For CNN Classifier

The CNN classifier directly processes pre-processed facial images rather than embeddings.

- **Input**: 160x160x3 RGB image (NumPy array).
- **Preprocessing**: Images are normalized to `[0, 1]` range.
- **Data Augmentation**: During training, data augmentation (random flips, rotations, zooms, brightness, contrast) is applied to the raw images to improve robustness.

#### Voice Embeddings (SpeechBrain)

**Model**: ECAPA-TDNN pre-trained on VoxCeleb (`speechbrain/spkrec-ecapa-voxceleb`)

- **Architecture**: ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation).
- **Embedding Dimension**: 192 features.
- **Preprocessing**:
  - Convert to mono if stereo.
  - Resample to 16kHz if necessary.
  - Amplitude normalization.

### Classifiers

The system implements two types of classifiers for both facial and voice recognition, selectable via configuration.

#### Facial Classifiers

1.  **Neural Network (NN) Classifier**:
    *   Operates on 512-dim FaceNet embeddings.
    *   Architecture: Fully-connected layers.
    *   Trained using `torch`.
2.  **Convolutional Neural Network (CNN) Classifier**:
    *   Operates directly on 160x160x3 raw facial images.
    *   Architecture: Custom CNN with residual blocks, batch normalization, and dropout.
    *   Trained using `tensorflow.keras`.

#### Voice Classifiers

1.  **SVM Classifier**:
    *   Operates on 192-dim SpeechBrain embeddings.
    *   Implemented using `scikit-learn`'s `SVC`.
2.  **Neural Network (NN) Classifier**:
    *   Operates on 192-dim SpeechBrain embeddings.
    *   Architecture: Fully-connected layers.
    *   Trained using `torch`.

## Training Models

### Facial Training Pipeline

The facial recognition models can be trained using either FaceNet embeddings with a traditional classifier (SVM or NN) or directly using a CNN on raw images.

#### Embedding-based Training (for SVM/NN classifiers)

1.  **Extract FaceNet Embeddings**: For each image in `models/facial/data/{username}/*.png`, a 512-dim embedding is extracted using `facenet-pytorch`'s `InceptionResnetV1` model.
2.  **Split Data**: Embeddings and corresponding labels are split into training and testing sets (e.g., 70/30 split).
3.  **Train Classifier**:
    *   **SVM**: An `SVC` model from `scikit-learn` is trained on the embeddings.
    *   **NN**: A fully-connected neural network is trained on the embeddings using `torch`.
4.  **Save Model**: The trained classifier (`.joblib` for SVM, `.pth` for NN) and its `LabelEncoder` are saved in `models/facial/checkpoints/`.

#### Image-based Training (for CNN classifier)

1.  **Load Raw Images**: Images from `models/facial/data/{username}/*.png` are loaded directly. Each image is expected to be a 160x160x3 RGB NumPy array.
2.  **Data Augmentation**: A `tensorflow.keras.Sequential` model applies various augmentations (flips, rotations, zooms, brightness, contrast) during training to the raw images.
3.  **Split Data**: Images and labels are split into training and validation sets.
4.  **Build and Train CNN**: A custom CNN model (`build_cnn_model` in `cnn_classifier.py`) is built using `tensorflow.keras` and trained on the raw image data, leveraging mixed precision training (FP16) if available. The training incorporates an early stopping mechanism and a Cosine Decay learning rate schedule.
5.  **Save Model**: The trained CNN model (`.keras` format) and its `LabelEncoder` are saved in `models/facial/checkpoints/`.

### Voice Training Pipeline

1.  **Extract SpeechBrain Embeddings**: For each audio segment in `models/voice/data/{username}/*.wav`, a 192-dim embedding is extracted using `speechbrain`'s `ECAPA-TDNN` model.
2.  **Split Data**: Embeddings and corresponding labels are split into training and testing sets.
3.  **Train Classifier**:
    *   **SVM**: An `SVC` model from `scikit-learn` is trained on the embeddings.
    *   **NN**: A fully-connected neural network is trained on the embeddings using `torch`.
4.  **Save Model**: The trained classifier (`.joblib` for SVM, `.pth` for NN) and its `LabelEncoder` are saved in `models/voice/checkpoints/`.

### Manual Training Commands

You can manually train the models using `Makefile` commands:

```bash
# Train facial models
make train-facial-svm    # Train with SVM on FaceNet embeddings
make train-facial-nn     # Train with Neural Network on FaceNet embeddings
make train-facial-cnn    # Train with CNN directly on images

# Train voice models
make train-voice-svm     # Train with SVM on SpeechBrain embeddings
make train-voice-nn     # Train with Neural Network on SpeechBrain embeddings
```

## Technologies Used

- **Backend**: Django, Django Channels
- **Machine Learning/Deep Learning**:
  - `tensorflow`: For building and training deep learning models (specifically CNN).
  - `torch`: For building and training deep learning models (FaceNet, SpeechBrain, NN classifiers).
  - `scikit-learn`: For machine learning algorithms (SVM classifiers).
  - `facenet-pytorch`: For facial embedding extraction.
  - `speechbrain`: For voice embedding extraction.
- **Database**: PostgreSQL
- **Other**: Docker, FFmpeg

## Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd facial-recognition
   ```
2. **Install FFmpeg**:
   ```bash
   # On macOS
   brew install ffmpeg

   # On Debian/Ubuntu
   sudo apt-get update && sudo apt-get install ffmpeg
   ```
3. **Set up the environment**:
   - Create a `.env` file from the `.env.example` and update it with your database credentials and other settings.
4. **Build and run the Docker containers**:
   ```bash
   docker-compose up -d --build
   ```
5. **Install the Python dependencies**:
    ```bash
    uv sync
    ```
6. **Apply the database migrations**:
   ```bash
   docker-compose exec web python src/manage.py migrate
   ```
7. **(Optional) Train initial models**:
    ```bash
    make train-facial-cnn
    make train-voice-nn
    ```

## Usage

1. **Start the Django development server**:
   ```bash
   docker-compose exec web python src/manage.py runserver 0.0.0.0:8000
   ```
2. **Access the web application** at `http://localhost:8000`.
3. **Register a new user**:
   - Go to the registration page and enter a username.
   - Record a 10-15 second video of yourself, speaking clearly.
   - The system will then train the models in the background.
4. **Log in**:
   - Go to the login page and enter your username.
   - Record a new video of yourself.
   - The system will verify your identity using facial and voice recognition.

## Future Work

- [ ] Liveness Detection
- [ ] Multi-factor authentication
- [ ] Video quality preprocessing
- [ ] Real-time feedback on frame quality
- [ ] Progressive model updates
- [ ] Support for multiple devices per user
- [ ] REST API for external integration
- [ ] Admin dashboard with metrics

---

**Document Version**: 1.3
**Last Updated**: 2024
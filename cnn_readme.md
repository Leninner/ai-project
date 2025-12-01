# CNN and NN Classifiers for Biometric Recognition

## Abstract

This document provides a technical overview of two key classifiers used in this biometric authentication system:
1. A **Convolutional Neural Network (CNN)** for facial recognition.
2. A **Neural Network (NN)** for voice recognition (speaker identification).

Both models are trained on data specific to this system, offering tailored identification capabilities.

---

## 1. Facial Recognition: CNN Classifier

### 1.1. Overview

The CNN classifier for facial recognition is designed for end-to-end learning, directly from preprocessed facial images. It removes the need for intermediate embedding extraction (like those from FaceNet), allowing the model to learn features optimized specifically for the individuals in the dataset.

- **Technology Stack**: TensorFlow/Keras
- **Input**: 160x160x3 pixel facial images.
- **Output**: A predicted identity and a confidence score.

### 1.2. Architecture

The CNN architecture is a custom-built model designed to balance performance and efficiency. It uses a series of residual blocks to learn hierarchical features from the face.

**Mermaid Diagram of Facial CNN Architecture:**

```mermaid
graph TD
    A[Input (160x160x3)] --> B(Initial Conv);
    B --> C{Residual Block 1 (32 filters)};
    C --> D[Max Pooling];
    D --> E{Residual Block 2 (64 filters)};
    E --> F[Max Pooling];
    F --> G{Residual Block 3 (128 filters)};
    G --> H[Max Pooling];
    H --> I[Global Average Pooling];
    I --> J[Dense Layers (256 -> 128)];
    J --> K[Output (Softmax)];
```

**Key Architectural Features:**
- **Residual Blocks**: Each block consists of two convolutional layers, batch normalization, and a skip connection. This helps prevent vanishing gradients and allows for deeper, more effective networks.
- **Progressive Downsampling**: The model uses `MaxPooling2D` to progressively reduce the spatial dimensions of the feature maps (160x160 -> 80x80 -> 40x40 -> 20x20).
- **Global Average Pooling**: Instead of a flatten layer, `GlobalAveragePooling2D` is used to reduce the feature maps to a single vector. This reduces the number of parameters and helps prevent overfitting.
- **Dense Layers**: A small number of dense layers at the end of the network perform the final classification.
- **Data Augmentation**: The model is trained with data augmentation (random flips, rotations, zooms) to improve its ability to generalize to new, unseen images.

**Simplified Layer Flow:**
1.  **Input** (160, 160, 3)
2.  **Initial Conv** -> (160, 160, 32)
3.  **Residual Block 1** (32 filters) + **Max Pooling** -> (80, 80, 32)
4.  **Residual Block 2** (64 filters) + **Max Pooling** -> (40, 40, 64)
5.  **Residual Block 3** (128 filters) + **Max Pooling** -> (20, 20, 128)
6.  **Global Average Pooling** -> (128,)
7.  **Dense Layers** -> (256,) -> (128,)
8.  **Output (Softmax)** -> (num_classes,)

### 1.3. Training Process

1.  **Data Loading**: Images are loaded from a directory structure where each subdirectory is a person's name. Images are expected to be preprocessed to 160x160x3.
2.  **Label Encoding**: String labels (names) are converted to integers.
3.  **Data Normalization**: Pixel values are scaled from [0, 255] to [0, 1].
4.  **Model Compilation**: The model is compiled with the Adam optimizer, `sparse_categorical_crossentropy` loss, and a `CosineDecay` learning rate schedule.
5.  **Callbacks**:
    - `EarlyStopping`: To stop training when validation loss stops improving.
    - `ModelCheckpoint`: To save the best performing model.
6.  **Fitting**: The model is trained using a generator that applies data augmentation on the fly. A portion of the data is held out for validation.

### 1.4. Inference Process

1.  **Model Loading**: The trained `.keras` model and the `LabelEncoder` are loaded from disk.
2.  **Image Preprocessing**: The input image is resized to 160x160, converted to a NumPy array, and normalized to the [0, 1] range.
3.  **Prediction**: The model predicts a probability distribution over the known classes.
4.  **Confidence Thresholding**: The highest probability is taken as the confidence score. If it is below a defined threshold (e.g., 0.75), the identity is considered "unknown".
5.  **Label Decoding**: If the confidence is high enough, the integer prediction is converted back to the person's name.

---

## 2. Voice Recognition: NN Classifier

### 2.1. Overview

The NN classifier for voice recognition operates on embeddings extracted from audio files. It uses a pre-trained model from SpeechBrain to generate these embeddings, which are then used to train a simple but effective neural network for speaker identification.

- **Technology Stack**: PyTorch, SpeechBrain
- **Input**: 192-dimensional embeddings from audio segments.
- **Output**: A predicted speaker identity and a confidence score.

### 2.2. Architecture

The voice classifier is a feed-forward neural network that takes the 192-dimension audio embedding as input.

**Mermaid Diagram of Voice NN Architecture:**
```mermaid
graph TD
    A[Input Embedding (192-dim)] --> B(Linear 128);
    B --> C(BatchNorm);
    C --> D(ReLU);
    D --> E(Dropout);
    E --> F(Linear 64);
    F --> G(BatchNorm);
    G --> H(ReLU);
    H --> I(Dropout);
    I --> J[Output (Linear)];
```
**Key Architectural Features:**
- **Pre-trained Embeddings**: The model leverages the powerful `speechbrain/spkrec-ecapa-voxceleb` model to extract a fixed-size (192-dim) vector representation of a voice snippet. This embedding captures the unique characteristics of a speaker's voice.
- **Fully Connected Layers**: The classifier itself is a simple sequence of linear layers.
- **Batch Normalization**: `BatchNorm1d` is used to stabilize training and improve generalization.
- **Dropout**: Dropout is applied between layers to prevent overfitting.
- **ReLU Activation**: The ReLU activation function is used to introduce non-linearity.

**Layer Flow:**
1.  **Input (Embedding)** (192,)
2.  **Linear** (192 -> 128) -> **BatchNorm** -> **ReLU** -> **Dropout**
3.  **Linear** (128 -> 64) -> **BatchNorm** -> **ReLU** -> **Dropout**
4.  **Output (Linear)** -> (num_classes,)

### 2.3. Training Process

1.  **Embedding Extraction**: For each `.wav` file in the dataset, a 192-dimensional embedding is extracted using the SpeechBrain encoder.
2.  **Data Loading**: The extracted embeddings and their corresponding labels are loaded.
3.  **Label Encoding**: String labels (names) are converted to integers.
4.  **Dataset & DataLoader**: The embeddings and labels are wrapped in a PyTorch `Dataset` and managed by a `DataLoader` for batching and shuffling.
5.  **Model Compilation**: The model uses the `CrossEntropyLoss` function and the Adam optimizer. A `ReduceLROnPlateau` learning rate scheduler is used to adjust the learning rate during training.
6.  **Training Loop**: The model is trained for a fixed number of epochs, iterating through the data in batches.

### 2.4. Inference Process

1.  **Model Loading**: The trained `.pth` model state and the `LabelEncoder` are loaded.
2.  **Embedding Extraction**: An embedding is extracted from the query audio file.
3.  **Prediction**: The model takes the query embedding and outputs logits. These are passed through a `softmax` function to get a probability distribution.
4.  **Confidence Thresholding**: The highest probability is taken as the confidence score. If it's below the threshold, the speaker is "unknown".
5.  **Label Decoding**: If confidence is sufficient, the predicted index is mapped back to the speaker's name.

---

## 3. Usage

Both classifiers are integrated into the system via a `ClassifierStrategy` pattern, which allows for easy swapping between different classification models.

### 3.1. Training via Pipeline

You can train the models using the provided pipeline scripts.

**Train CNN Facial Classifier:**
```bash
python -m models.facial.pipeline --classifier cnn
```

**Train NN Voice Classifier:**
```bash
python -m models.voice.pipeline --classifier nn
```

### 3.2. Direct Python Usage

#### Facial Recognition
```python
from models.facial.classifiers.classifier_strategy import create_classifier_strategy, FaceIdentifier

# Create CNN strategy
strategy = create_classifier_strategy('cnn')
face_identifier = FaceIdentifier(strategy)

# Train
face_identifier.train(data_dir='models/facial/data')

# Identify
person, confidence = face_identifier.identify_face(image_path='path/to/face.jpg')
print(f"Identified: {person} with {confidence:.2%} confidence")
```

#### Voice Recognition
```python
from models.voice.classifiers.classifier_strategy import create_classifier_strategy, VoiceIdentifier

# Create NN strategy
strategy = create_classifier_strategy('nn')
voice_identifier = VoiceIdentifier(strategy)

# Train
voice_identifier.train(data_dir='models/voice/data')

# Identify
speaker, confidence = voice_identifier.identify_speaker(audio_path='path/to/voice.wav')
print(f"Identified: {speaker} with {confidence:.2%} confidence")
```

import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import sys

DATA_DIR = Path(__file__).parent.parent / "data"
CHECKPOINT_PATH = Path(__file__).parent.parent / "checkpoints"
CNN_MODEL_PATH = CHECKPOINT_PATH / "cnn_classifier.keras"
LABEL_ENCODER_PATH = CHECKPOINT_PATH / "cnn_label_encoder.pkl"
CONFIDENCE_THRESHOLD = 0.65

IMAGE_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

_cnn_model = None
_label_encoder = None

tf.config.set_soft_device_placement(True)
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU devices available: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
    else:
        print("No GPU devices found, using CPU")
except Exception as e:
    print(f"GPU configuration warning: {e}")

try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision training enabled (FP16)")
except Exception as e:
    print(f"Mixed precision not available: {e}")

def _get_preprocessor():
    facial_path = Path(__file__).parent.parent
    if str(facial_path) not in sys.path:
        sys.path.insert(0, str(facial_path))
    from facial_preprocessor import FacialPreprocessor
    return FacialPreprocessor()

def preprocess_image_for_training(image_path):
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image, dtype=np.uint8)
    if image_array.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"Image {image_path} has incorrect shape: {image_array.shape}. Expected ({IMAGE_SIZE}, {IMAGE_SIZE}, 3)")
    return image_array

def load_images_from_directory(data_dir):
    images = []
    labels = []
    
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    elif not isinstance(data_dir, Path):
        data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        for image_name in os.listdir(person_dir):
            if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            image_path = os.path.join(person_dir, image_name)
            try:
                image_array = preprocess_image_for_training(image_path)
                images.append(image_array)
                labels.append(person_name)
            except Exception as e:
                continue
    
    return np.array(images), np.array(labels)

def build_cnn_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def create_data_augmentation():
    return keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomFlip("horizontal"),
        layers.RandomBrightness(0.1),
        layers.RandomContrast(0.1),
    ])

def train_cnn_classifier(data_dir=None, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE):
    CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
    
    if data_dir is None:
        data_dir = DATA_DIR
    elif isinstance(data_dir, str):
        data_dir = Path(data_dir)
    elif not isinstance(data_dir, Path):
        data_dir = Path(data_dir)
    
    print(f"Loading images from {data_dir}...")
    images, labels = load_images_from_directory(data_dir)
    
    if len(images) == 0:
        raise ValueError(f"No images found in {data_dir}")
    
    print(f"Loaded {len(images)} images from {len(np.unique(labels))} classes")
    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    
    images = images.astype(np.float32) / 255.0
    
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    model = build_cnn_model(input_shape, num_classes)
    
    data_augmentation = create_data_augmentation()
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
    
    model_checkpoint = callbacks.ModelCheckpoint(
        str(CNN_MODEL_PATH),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    from sklearn.model_selection import train_test_split
    
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    print(f"Training CNN classifier...")
    print(f"Classes: {num_classes}, Epochs: {epochs}, Batch size: {batch_size}")
    print(f"Input shape: {input_shape}")
    print(f"Training samples: {len(train_images)}, Validation samples: {len(val_images)}")
    
    def augmented_generator(images, labels, batch_size, augment=True):
        while True:
            indices = np.random.permutation(len(images))
            for i in range(0, len(images), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_images = images[batch_indices]
                batch_labels = labels[batch_indices]
                
                if augment:
                    batch_images_augmented = data_augmentation(batch_images, training=True)
                else:
                    batch_images_augmented = batch_images
                yield batch_images_augmented, batch_labels
    
    train_gen = augmented_generator(train_images, train_labels, batch_size, augment=True)
    val_gen = augmented_generator(val_images, val_labels, batch_size, augment=False)
    
    steps_per_epoch = len(train_images) // batch_size
    validation_steps = len(val_images) // batch_size
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )
    
    model.save(str(CNN_MODEL_PATH))
    
    import pickle
    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"CNN model trained and saved to {CNN_MODEL_PATH}")
    
    return model, label_encoder

def load_cnn_classifier():
    global _cnn_model, _label_encoder
    
    if _cnn_model is None or _label_encoder is None:
        if CNN_MODEL_PATH.exists() and LABEL_ENCODER_PATH.exists():
            import pickle
            with open(LABEL_ENCODER_PATH, 'rb') as f:
                _label_encoder = pickle.load(f)
            
            _cnn_model = keras.models.load_model(str(CNN_MODEL_PATH))
        else:
            raise ValueError("CNN classifier not trained. Please train the model first.")
    
    return _cnn_model, _label_encoder

def identify_face_from_image(image_path):
    model, label_encoder = load_cnn_classifier()
    
    try:
        image = Image.open(image_path).convert('RGB')
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        
        face_array = np.array(image, dtype=np.uint8)
        
        if face_array.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
            raise ValueError(f"Image has incorrect shape: {face_array.shape}. Expected ({IMAGE_SIZE}, {IMAGE_SIZE}, 3)")
        
        face_array = face_array.astype(np.float32) / 255.0
        face_array = np.expand_dims(face_array, axis=0)
        
        predictions = model.predict(face_array, verbose=0)
        probabilities = predictions[0]
        
        predicted_index = np.argmax(probabilities)
        confidence_score = float(probabilities[predicted_index])
        
        if confidence_score < CONFIDENCE_THRESHOLD:
            return "unknown", confidence_score
        
        identified_person = label_encoder.inverse_transform([predicted_index])[0]
        return identified_person, confidence_score
    
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

def identify_face_from_array(face_array):
    model, label_encoder = load_cnn_classifier()
    
    if len(face_array.shape) == 3:
        if face_array.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
            image = Image.fromarray(face_array.astype(np.uint8) if face_array.dtype != np.uint8 else face_array)
            if image.size != (IMAGE_SIZE, IMAGE_SIZE):
                image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
            face_array = np.array(image, dtype=np.uint8)
        
        face_array = np.expand_dims(face_array, axis=0)
    
    if face_array.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"Expected image shape ({IMAGE_SIZE}, {IMAGE_SIZE}, 3), got {face_array.shape[1:]}")
    
    if face_array.dtype != np.float32:
        if face_array.dtype == np.uint8:
            face_array = face_array.astype(np.float32) / 255.0
        else:
            face_array = face_array.astype(np.float32)
            if face_array.max() > 1.0:
                face_array = face_array / 255.0
    
    predictions = model.predict(face_array, verbose=0)
    probabilities = predictions[0]
    
    predicted_index = np.argmax(probabilities)
    confidence_score = float(probabilities[predicted_index])
    
    if confidence_score < CONFIDENCE_THRESHOLD:
        return "unknown", confidence_score
    
    identified_person = label_encoder.inverse_transform([predicted_index])[0]
    return identified_person, confidence_score

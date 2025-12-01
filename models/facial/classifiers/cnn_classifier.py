import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.callbacks import TensorBoard
import sys

DATA_DIR = Path(__file__).parent.parent / "data"
CHECKPOINT_PATH = Path(__file__).parent.parent / "checkpoints"
CNN_MODEL_PATH = CHECKPOINT_PATH / "cnn_classifier.keras"
LABEL_ENCODER_PATH = CHECKPOINT_PATH / "cnn_label_encoder.pkl"
CONFIDENCE_THRESHOLD = 0.65

IMAGE_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.003

_cnn_model = None
_label_encoder = None

tf.config.set_soft_device_placement(True)
try:
    gpus = tf.config.list_physical_devices("GPU")
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
    policy = tf.keras.mixed_precision.Policy("mixed_float16")
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
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image, dtype=np.uint8)
    if image_array.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(
            f"Image {image_path} has incorrect shape: {image_array.shape}. Expected ({IMAGE_SIZE}, {IMAGE_SIZE}, 3)"
        )

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
            image_path = os.path.join(person_dir, image_name)

            try:
                image_array = preprocess_image_for_training(image_path)
                images.append(image_array)
                labels.append(person_name)
            except Exception:
                continue

    return np.array(images), np.array(labels)


def conv_block(x, filters, kernel_size=(3, 3), strides=(1, 1), dropout_rate=0.2):
    """Convolutional block with batch normalization and dropout"""
    conv = layers.Conv2D(filters, kernel_size, strides=strides, padding="same")(x)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation("relu")(conv)
    conv = layers.Conv2D(filters, kernel_size, padding="same")(conv)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation("relu")(conv)
    conv = layers.SpatialDropout2D(dropout_rate)(conv)
    return conv


def residual_block(x, filters, dropout_rate=0.2):
    """Residual-like block with skip connection"""
    shortcut = x

    # Main path
    x = conv_block(x, filters, dropout_rate=dropout_rate)

    # Match dimensions if needed
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Add skip connection
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x


def build_cnn_model(input_shape, num_classes):
    """
    Simplified CNN model with 3 convolutional blocks

    Architecture:
    - Input: 160x160x3
    - Block 1: 32 filters -> 80x80x32
    - Block 2: 64 filters -> 40x40x64
    - Block 3: 128 filters -> 20x20x128
    - Global Average Pooling -> 128
    - Dense: 256 -> 128 -> num_classes

    This architecture preserves more spatial information by stopping at 20x20
    instead of going down to 10x10, which was too pixelated.
    """
    inputs = layers.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Block 1: 32 filters (160x160 -> 80x80)
    x = residual_block(x, 32, dropout_rate=0.1)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2: 64 filters (80x80 -> 40x40)
    x = residual_block(x, 64, dropout_rate=0.15)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3: 128 filters (40x40 -> 20x20)
    x = residual_block(x, 128, dropout_rate=0.2)
    x = layers.MaxPooling2D((2, 2))(x)

    # Global pooling (20x20x128 -> 128)
    x = layers.GlobalAveragePooling2D()(x)

    # Simplified dense layers
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


def create_data_augmentation():
    """Create data augmentation pipeline with moderate values for facial recognition"""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),  # ~18° rotation (natural head tilt)
            layers.RandomZoom(0.05),  # Slight zoom variations
            layers.RandomTranslation(0.05, 0.05),  # Minor position shifts
            layers.RandomBrightness(0.05),  # Realistic lighting changes
            layers.RandomContrast(0.05),  # Subtle contrast variations
        ]
    )


def plot_training_history(history, use_augmentation=True):
    """Plot and save training history (loss and accuracy curves)"""
    import matplotlib.pyplot as plt

    aug_suffix = "_with_aug" if use_augmentation else "_no_aug"

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot accuracy
    ax1.plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
    ax1.set_title(
        f"Model Accuracy {'(With Augmentation)' if use_augmentation else '(No Augmentation)'}",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.legend(loc="lower right", fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot loss
    ax2.plot(history.history["loss"], label="Train Loss", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
    ax2.set_title(
        f"Model Loss {'(With Augmentation)' if use_augmentation else '(No Augmentation)'}",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.legend(loc="upper right", fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = CHECKPOINT_PATH / f"training_history{aug_suffix}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Training history plot saved: {plot_path}")
    plt.close()


def train_cnn_classifier(
    data_dir=None,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    use_augmentation=True,
):
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

    initial_learning_rate = learning_rate

    # Cosine decay learning rate schedule
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=epochs * max(1, len(images) // batch_size),
        alpha=0.1,  # Minimum learning rate will be 10% of initial
    )

    optimizer = keras.optimizers.Adam(
        learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-7
    )
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        verbose=1,
        min_delta=1e-6,
    )

    # Note: ReduceLROnPlateau removed - using CosineDecay schedule instead

    model_checkpoint = callbacks.ModelCheckpoint(
        str(CNN_MODEL_PATH), monitor="val_loss", save_best_only=True, verbose=1
    )

    from sklearn.model_selection import train_test_split

    train_images, val_images, train_labels, val_labels = train_test_split(
        images, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )

    print("Training CNN classifier...")
    print(f"Classes: {num_classes}, Epochs: {epochs}, Batch size: {batch_size}")
    print(f"Input shape: {input_shape}")
    print(f"Data augmentation: {'ENABLED' if use_augmentation else 'DISABLED'}")
    print(
        f"Training samples: {len(train_images)}, Validation samples: {len(val_images)}"
    )

    def augmented_generator(images, labels, batch_size, augment=True, shuffle=True):
        """Generator with optional shuffling for training/validation"""
        while True:
            # Only shuffle for training, not validation
            if shuffle:
                indices = np.random.permutation(len(images))
            else:
                indices = np.arange(len(images))

            for i in range(0, len(images), batch_size):
                batch_indices = indices[i : i + batch_size]
                batch_images = images[batch_indices]
                batch_labels = labels[batch_indices]

                if augment:
                    batch_images_augmented = data_augmentation(
                        batch_images, training=True
                    )
                else:
                    batch_images_augmented = batch_images
                yield batch_images_augmented, batch_labels

    train_gen = augmented_generator(
        train_images, train_labels, batch_size, augment=use_augmentation, shuffle=True
    )
    # IMPORTANT: Don't shuffle validation data for consistent metrics
    val_gen = augmented_generator(
        val_images, val_labels, batch_size, augment=False, shuffle=False
    )

    steps_per_epoch = max(1, len(train_images) // batch_size)
    validation_steps = max(1, len(val_images) // batch_size)

    print(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=validation_steps,
        verbose=1,
        callbacks=[
            early_stopping,
            model_checkpoint,
            TensorBoard(log_dir="logs/fit"),
        ],
    )

    final_train_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]
    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]

    print("\nTraining completed:")
    print(
        f"  Final training loss: {final_train_loss:.4f}, accuracy: {final_train_acc:.4f}"
    )
    print(
        f"  Final validation loss: {final_val_loss:.4f}, accuracy: {final_val_acc:.4f}"
    )

    # Plot training history
    print("\nGenerating training history plots...")
    plot_training_history(history, use_augmentation)

    model.save(str(CNN_MODEL_PATH))

    import pickle

    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"CNN model trained and saved to {CNN_MODEL_PATH}")

    return model, label_encoder


def load_cnn_classifier():
    global _cnn_model, _label_encoder

    if _cnn_model is None or _label_encoder is None:
        if CNN_MODEL_PATH.exists() and LABEL_ENCODER_PATH.exists():
            import pickle

            with open(LABEL_ENCODER_PATH, "rb") as f:
                _label_encoder = pickle.load(f)

            _cnn_model = keras.models.load_model(str(CNN_MODEL_PATH))

            num_classes = len(_label_encoder.classes_)
            print(
                f"Loaded CNN model with {num_classes} classes: {list(_label_encoder.classes_)}"
            )
        else:
            raise ValueError(
                "CNN classifier not trained. Please train the model first."
            )

    return _cnn_model, _label_encoder


def identify_face_from_image(image_path):
    model, label_encoder = load_cnn_classifier()

    try:
        image = Image.open(image_path).convert("RGB")
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)

        face_array = np.array(image, dtype=np.uint8)

        if face_array.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
            raise ValueError(
                f"Image has incorrect shape: {face_array.shape}. Expected ({IMAGE_SIZE}, {IMAGE_SIZE}, 3)"
            )

        face_array = face_array.astype(np.float32) / 255.0
        face_array = np.expand_dims(face_array, axis=0)

        predictions = model.predict(face_array, verbose=0)
        # log probabilities in a good wat
        print(f"---Probabilities: {predictions}")
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

    original_shape = face_array.shape
    original_dtype = face_array.dtype

    if len(face_array.shape) == 3:
        if face_array.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
            image = Image.fromarray(
                face_array.astype(np.uint8)
                if face_array.dtype != np.uint8
                else face_array
            )
            if image.size != (IMAGE_SIZE, IMAGE_SIZE):
                image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
            face_array = np.array(image, dtype=np.uint8)

        face_array = np.expand_dims(face_array, axis=0)

    if face_array.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(
            f"Expected image shape ({IMAGE_SIZE}, {IMAGE_SIZE}, 3), got {face_array.shape[1:]}"
        )

    if face_array.dtype == np.uint8:
        face_array = face_array.astype(np.float32) / 255.0
    elif face_array.dtype != np.float32:
        face_array = face_array.astype(np.float32)
        if face_array.max() > 1.0:
            face_array = face_array / 255.0
    elif face_array.max() > 1.0:
        face_array = face_array / 255.0

    if face_array.min() < 0.0 or face_array.max() > 1.0:
        face_array = np.clip(face_array, 0.0, 1.0)

    predictions = model.predict(face_array, verbose=0)
    probabilities = predictions[0]

    if not np.allclose(probabilities.sum(), 1.0, atol=1e-5):
        probabilities = probabilities / probabilities.sum()

    predicted_index = np.argmax(probabilities)
    confidence_score = float(probabilities[predicted_index])

    if confidence_score < CONFIDENCE_THRESHOLD:
        return "unknown", confidence_score

    identified_person = label_encoder.inverse_transform([predicted_index])[0]
    return identified_person, confidence_score

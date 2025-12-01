import os
import numpy as np
import pickle
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

# Try to import keras serialization decorator (Keras 3 feature)
try:
    from keras.saving import register_keras_serializable
    HAS_KERAS_SERIALIZABLE = True
except (ImportError, AttributeError):
    # Fallback for TensorFlow's bundled Keras (Keras 2.x)
    def register_keras_serializable():
        """Dummy decorator for compatibility"""
        def decorator(cls):
            return cls
        return decorator
    HAS_KERAS_SERIALIZABLE = False

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
CHECKPOINT_PATH = Path(__file__).parent.parent / "checkpoints"
EMBEDDING_MODEL_PATH = CHECKPOINT_PATH / "embedding_model.keras"
EMBEDDING_DATABASE_PATH = CHECKPOINT_PATH / "embedding_database.pkl"
SIMILARITY_THRESHOLD = 0.7  # Increased for better embeddings trained with triplet loss
IMAGE_SIZE = 160

_embedding_model = None
_embedding_database = None

# Enable unsafe deserialization (safe for our own model files)
# Only available in Keras 3
try:
    keras.config.enable_unsafe_deserialization()
except AttributeError:
    pass  # Not available in older Keras versions

# GPU Configuration (same as cnn_classifier)
tf.config.set_soft_device_placement(True)
try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU devices available: {len(gpus)}")
except Exception as e:
    print(f"GPU configuration warning: {e}")


def conv_block(x, filters, kernel_size=(3, 3), strides=(1, 1), dropout_rate=0.2):
    """Convolutional block with batch normalization and dropout"""
    conv = layers.Conv2D(
        filters, 
        kernel_size, 
        strides=strides, 
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation("relu")(conv)
    conv = layers.Conv2D(
        filters, 
        kernel_size, 
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(conv)
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
        shortcut = layers.Conv2D(
            filters, 
            (1, 1), 
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Add skip connection
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x


@register_keras_serializable()
class L2Normalization(layers.Layer):
    """Custom L2 normalization layer that's serializable"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)
    
    def get_config(self):
        config = super().get_config()
        return config


def build_embedding_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)):
    """
    Build enhanced CNN model that outputs L2-normalized 256-dim embeddings.
    
    This is the improved architecture designed for triplet loss training.
    
    Architecture:
    - Input: 160x160x3
    - Block 1: 32 filters -> 80x80x32
    - Block 2: 64 filters -> 40x40x64
    - Block 3: 128 filters -> 20x20x128
    - Block 4: 256 filters -> 10x10x256  [ENHANCED]
    - Global Average Pooling -> 256
    - Dense: 512 -> 256 (embeddings)
    - L2 Normalization -> 256 (unit vector)
    """
    inputs = layers.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv2D(
        32, 
        (3, 3), 
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(inputs)
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

    # Block 4: 256 filters (20x20 -> 10x10) [ENHANCED - deeper network]
    x = residual_block(x, 256, dropout_rate=0.25)
    x = layers.MaxPooling2D((2, 2))(x)

    # Global pooling (10x10x256 -> 256)
    x = layers.GlobalAveragePooling2D()(x)

    # Dense layers for embeddings
    x = layers.Dense(
        512, 
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Embedding layer (256-dim, increased from 128)
    x = layers.Dense(
        256, 
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # L2 normalization - makes embeddings unit vectors
    # This allows cosine similarity to be computed as a simple dot product
    embeddings = L2Normalization()(x)

    model = models.Model(inputs=inputs, outputs=embeddings)
    return model


def load_embedding_model():
    """Load the trained embedding model"""
    global _embedding_model
    
    if _embedding_model is None:
        if not EMBEDDING_MODEL_PATH.exists():
            raise ValueError(
                f"Embedding model not found at {EMBEDDING_MODEL_PATH}. "
                "Please train the model first or convert an existing classifier."
            )
        
        # Load with custom objects to handle L2Normalization layer and TripletLoss
        _embedding_model = keras.models.load_model(
            str(EMBEDDING_MODEL_PATH),
            custom_objects={
                "L2Normalization": L2Normalization,
                "TripletLoss": TripletLoss
            }
        )
        print(f"Loaded embedding model from {EMBEDDING_MODEL_PATH}")
    
    return _embedding_model


def load_embedding_database():
    """Load the embedding database containing reference embeddings for each person"""
    global _embedding_database
    
    if _embedding_database is None:
        if not EMBEDDING_DATABASE_PATH.exists():
            raise ValueError(
                f"Embedding database not found at {EMBEDDING_DATABASE_PATH}. "
                "Please generate the database first."
            )
        
        with open(EMBEDDING_DATABASE_PATH, "rb") as f:
            _embedding_database = pickle.load(f)
        
        print(f"Loaded embedding database with {len(_embedding_database)} persons")
        print(f"  Persons: {list(_embedding_database.keys())}")
    
    return _embedding_database


def extract_embedding(image_path):
    """
    Extract 128-dim L2-normalized embedding from an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        numpy array of shape (128,) containing the normalized embedding
    """
    model = load_embedding_model()
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    if image.size != (IMAGE_SIZE, IMAGE_SIZE):
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    
    image_array = np.array(image, dtype=np.uint8)
    
    if image_array.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(
            f"Image has incorrect shape: {image_array.shape}. "
            f"Expected ({IMAGE_SIZE}, {IMAGE_SIZE}, 3)"
        )
    
    # Normalize to [0, 1]
    image_array = image_array.astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Extract embedding
    embedding = model.predict(image_array, verbose=0)
    return embedding[0]  # Return as 1D array


def extract_embedding_from_array(face_array):
    """
    Extract embedding from a numpy array.
    
    Args:
        face_array: numpy array of shape (H, W, 3) or (1, H, W, 3)
        
    Returns:
        numpy array of shape (128,) containing the normalized embedding
    """
    model = load_embedding_model()
    
    # Handle different input shapes
    if len(face_array.shape) == 3:
        if face_array.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
            image = Image.fromarray(
                face_array.astype(np.uint8) if face_array.dtype != np.uint8 else face_array
            )
            if image.size != (IMAGE_SIZE, IMAGE_SIZE):
                image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
            face_array = np.array(image, dtype=np.uint8)
        
        face_array = np.expand_dims(face_array, axis=0)
    
    if face_array.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(
            f"Expected image shape ({IMAGE_SIZE}, {IMAGE_SIZE}, 3), got {face_array.shape[1:]}"
        )
    
    # Normalize
    if face_array.dtype == np.uint8:
        face_array = face_array.astype(np.float32) / 255.0
    elif face_array.dtype != np.float32:
        face_array = face_array.astype(np.float32)
        if face_array.max() > 1.0:
            face_array = face_array / 255.0
    elif face_array.max() > 1.0:
        face_array = face_array / 255.0
    
    face_array = np.clip(face_array, 0.0, 1.0)
    
    # Extract embedding
    embedding = model.predict(face_array, verbose=0)
    return embedding[0]


def compute_cosine_similarity(embedding1, embedding2):
    """
    Compute cosine similarity between two embeddings.
    
    Since embeddings are L2-normalized, cosine similarity is simply the dot product.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        float: Similarity score in range [-1, 1], where 1 = identical, -1 = opposite
    """
    # Ensure inputs are numpy arrays
    emb1 = np.array(embedding1)
    emb2 = np.array(embedding2)
    
    # Cosine similarity for normalized vectors is just the dot product
    similarity = np.dot(emb1, emb2)
    
    return float(similarity)


def identify_face_from_image(image_path, expected_name=None):
    """
    Identify a person from an image using embedding comparison.
    
    Args:
        image_path: Path to the face image
        expected_name: Optional expected person name for targeted authentication.
                      If None, performs general identification (finds best match).
        
    Returns:
        Tuple of (person_name, similarity_score)
        - Authentication mode (expected_name provided):
          Returns (expected_name, score) if match found and similarity >= threshold
          Returns ("unknown", score) if similarity < threshold or user not found
        - Identification mode (expected_name is None):
          Returns (best_match_name, score) if similarity >= threshold
          Returns ("unknown", score) if no match above threshold
    """
    try:
        # Extract embedding from query image
        query_embedding = extract_embedding(image_path)
        
        # Load reference embeddings
        database = load_embedding_database()
        
        if len(database) == 0:
            return "unknown", 0.0
        
        if expected_name is not None:
            # AUTHENTICATION MODE: Targeted validation against expected user
            expected_name_normalized = expected_name.lower().strip()
            
            if expected_name_normalized not in database:
                print(f"Expected user '{expected_name_normalized}' not found in database")
                print(f"Available users: {list(database.keys())}")
                return "unknown", 0.0
            
            # Compare only with the expected user's embedding
            ref_embedding = database[expected_name_normalized]
            similarity = compute_cosine_similarity(query_embedding, ref_embedding)
            
            print(f"[AUTH MODE] Checking against expected user '{expected_name_normalized}': similarity = {similarity:.4f}")
            
            # Check if similarity meets threshold
            if similarity >= SIMILARITY_THRESHOLD:
                return expected_name_normalized, float(similarity)
            else:
                return "unknown", float(similarity)
        else:
            # IDENTIFICATION MODE: Find best match among all users
            best_match = None
            best_similarity = -1.0
            
            for person_name, ref_embedding in database.items():
                similarity = compute_cosine_similarity(query_embedding, ref_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = person_name
            
            print(f"[ID MODE] Best match: '{best_match}' with similarity = {best_similarity:.4f}")
            
            # Check if similarity meets threshold
            if best_similarity >= SIMILARITY_THRESHOLD:
                return best_match, float(best_similarity)
            else:
                return "unknown", float(best_similarity)
        
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")


def identify_face_from_array(face_array, expected_name=None):
    """
    Identify a person from a face array using embedding comparison.
    
    Args:
        face_array: numpy array of the face image
        expected_name: Optional expected person name for targeted authentication.
                      If None, performs general identification (finds best match).
        
    Returns:
        Tuple of (person_name, similarity_score)
        - Authentication mode (expected_name provided):
          Returns (expected_name, score) if match found and similarity >= threshold
          Returns ("unknown", score) if similarity < threshold or user not found
        - Identification mode (expected_name is None):
          Returns (best_match_name, score) if similarity >= threshold
          Returns ("unknown", score) if no match above threshold
    """
    try:
        # Extract embedding from query array
        query_embedding = extract_embedding_from_array(face_array)
        
        # Load reference embeddings
        database = load_embedding_database()
        
        if len(database) == 0:
            return "unknown", 0.0
        
        if expected_name is not None:
            # AUTHENTICATION MODE: Targeted validation against expected user
            expected_name_normalized = expected_name.lower().strip()
            
            if expected_name_normalized not in database:
                print(f"Expected user '{expected_name_normalized}' not found in database")
                print(f"Available users: {list(database.keys())}")
                return "unknown", 0.0
            
            # Compare only with the expected user's embedding
            ref_embedding = database[expected_name_normalized]
            similarity = compute_cosine_similarity(query_embedding, ref_embedding)
            
            print(f"[AUTH MODE] Checking against expected user '{expected_name_normalized}': similarity = {similarity:.4f}")
            
            # Check if similarity meets threshold
            if similarity >= SIMILARITY_THRESHOLD:
                return expected_name_normalized, float(similarity)
            else:
                return "unknown", float(similarity)
        else:
            # IDENTIFICATION MODE: Find best match among all users
            best_match = None
            best_similarity = -1.0
            
            for person_name, ref_embedding in database.items():
                similarity = compute_cosine_similarity(query_embedding, ref_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = person_name
            
            print(f"[ID MODE] Best match: '{best_match}' with similarity = {best_similarity:.4f}")
            
            # Check if similarity meets threshold
            if best_similarity >= SIMILARITY_THRESHOLD:
                return best_match, float(best_similarity)
            else:
                return "unknown", float(best_similarity)
        
    except Exception as e:
        raise ValueError(f"Error processing face array: {str(e)}")


def save_embedding_database(data_dir=None, model=None):
    """
    Generate and save embedding database from training data.
    
    For each person in the data directory:
    1. Extract embeddings from all their images
    2. Compute the average embedding
    3. Save to database
    
    Args:
        data_dir: Path to directory containing person subdirectories
        model: Optional pre-loaded model (loads if not provided)
    """
    if data_dir is None:
        data_dir = DATA_DIR
    elif isinstance(data_dir, str):
        data_dir = Path(data_dir)
    
    if model is None:
        model = load_embedding_model()
    
    print(f"Generating embedding database from {data_dir}...")
    
    database = {}
    
    # Iterate through each person's directory
    for person_name in os.listdir(data_dir):
        person_dir = data_dir / person_name
        
        if not person_dir.is_dir():
            continue
        
        embeddings = []
        
        # Extract embeddings for all images of this person
        for image_name in os.listdir(person_dir):
            image_path = person_dir / image_name
            
            try:
                embedding = extract_embedding(str(image_path))
                embeddings.append(embedding)
            except Exception as e:
                print(f"  Warning: Could not process {image_path}: {e}")
                continue
        
        if len(embeddings) == 0:
            print(f"  Warning: No valid images found for {person_name}")
            continue
        
        # Compute average embedding
        avg_embedding = np.mean(embeddings, axis=0)
        
        # Re-normalize (averaging may slightly denormalize)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        database[person_name] = avg_embedding
        
        print(f"  {person_name}: {len(embeddings)} images -> avg embedding")
    
    # Save database
    CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
    with open(EMBEDDING_DATABASE_PATH, "wb") as f:
        pickle.dump(database, f)
    
    print(f"\nEmbedding database saved to {EMBEDDING_DATABASE_PATH}")
    print(f"Total persons: {len(database)}")
    print(f"Persons: {list(database.keys())}")
    
    return database


def convert_cnn_classifier_to_embedding_model():
    """
    Convert the existing trained CNN classifier to an embedding model.
    
    This removes the final softmax layer and uses the 128-dim dense layer as embeddings.
    """
    from .cnn_classifier import CNN_MODEL_PATH as CLASSIFIER_MODEL_PATH
    
    if not CLASSIFIER_MODEL_PATH.exists():
        raise ValueError(
            f"CNN classifier not found at {CLASSIFIER_MODEL_PATH}. "
            "Please train the classifier first."
        )
    
    print(f"Loading classifier from {CLASSIFIER_MODEL_PATH}...")
    classifier = keras.models.load_model(str(CLASSIFIER_MODEL_PATH))
    
    print("Converting to embedding model...")
    
    # Find the layer before the final softmax (should be the 128-dim dense layer)
    # The architecture is: ... -> Dense(128) -> BatchNorm -> Dropout -> Dense(num_classes, softmax)
    # We want to get output after the second Dense(128) + BatchNorm + Dropout
    
    # Get the layer that outputs to the final classification layer
    embedding_layer = classifier.layers[-2]  # Layer before final Dense
    
    # Create new model with same input but output at embedding layer
    embedding_model = models.Model(
        inputs=classifier.input,
        outputs=embedding_layer.output
    )
    
    # Add L2 normalization using the custom layer
    normalized_model = models.Model(
        inputs=embedding_model.input,
        outputs=L2Normalization()(embedding_model.output)
    )
    
    # Save the embedding model
    CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
    normalized_model.save(str(EMBEDDING_MODEL_PATH))
    
    print(f"Embedding model saved to {EMBEDDING_MODEL_PATH}")
    print(f"Model input shape: {normalized_model.input_shape}")
    print(f"Model output shape: {normalized_model.output_shape}")
    
    return normalized_model


# ============================================================================
# Triplet Loss Training Components
# ============================================================================

class TripletBatchGenerator:
    """
    Generator that yields batches of triplets (anchor, positive, negative).
    
    Each batch contains P persons with K images each, allowing for 
    online triplet mining within the batch.
    """
    
    def __init__(self, images, labels, persons_per_batch, images_per_person, augmentation=None):
        """
        Args:
            images: numpy array of images (N, H, W, C)
            labels: numpy array of labels (N,) - can be strings or ints
            persons_per_batch: Number of different persons in each batch (P)
            images_per_person: Number of images per person in each batch (K)
            augmentation: Optional augmentation pipeline
        """
        self.images = images
        self.labels = labels
        self.persons_per_batch = persons_per_batch
        self.images_per_person = images_per_person
        self.augmentation = augmentation
        
        # Create label encoder for string labels
        self.unique_labels = np.unique(labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        
        # Group images by person
        self.person_to_images = {}
        for idx, label in enumerate(labels):
            if label not in self.person_to_images:
                self.person_to_images[label] = []
            self.person_to_images[label].append(idx)
        
        # Filter persons with enough images
        self.valid_persons = [
            person for person, indices in self.person_to_images.items()
            if len(indices) >= images_per_person
        ]
        
        if len(self.valid_persons) < persons_per_batch:
            raise ValueError(
                f"Not enough persons with {images_per_person}+ images. "
                f"Found {len(self.valid_persons)}, need {persons_per_batch}"
            )
        
        print(f"TripletBatchGenerator: {len(self.valid_persons)} valid persons")
    
    def __call__(self):
        """Generate batches indefinitely"""
        while True:
            # Sample P persons randomly
            selected_persons = np.random.choice(
                self.valid_persons, 
                self.persons_per_batch, 
                replace=False
            )
            
            batch_images = []
            batch_labels = []
            
            # For each person, sample K images
            for person in selected_persons:
                person_indices = self.person_to_images[person]
                selected_indices = np.random.choice(
                    person_indices,
                    self.images_per_person,
                    replace=False
                )
                
                for idx in selected_indices:
                    batch_images.append(self.images[idx])
                    # Convert label to integer
                    batch_labels.append(self.label_to_idx[self.labels[idx]])
            
            batch_images = np.array(batch_images, dtype=np.float32)
            batch_labels = np.array(batch_labels, dtype=np.int32)
            
            # Apply augmentation if provided
            if self.augmentation is not None:
                batch_images = self.augmentation(batch_images, training=True)
            
            yield batch_images, batch_labels


@register_keras_serializable()
class TripletLoss(keras.losses.Loss):
    """
    Triplet loss with semi-hard negative mining.
    
    For each anchor-positive pair, we select the hardest negative that is:
    - Further from anchor than positive (valid negative)
    - Within the margin (semi-hard)
    
    Loss = max(0, ||anchor - positive||² - ||anchor - negative||² + margin)
    """
    
    def __init__(self, margin=0.3, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
    
    def call(self, y_true, y_pred):
        """
        Args:
            y_true: Labels (batch_size,)
            y_pred: Embeddings (batch_size, embedding_dim)
        """
        # Compute pairwise distances
        embeddings = y_pred
        
        # Pairwise squared distances using L2-normalized embeddings
        # ||a - b||² = 2 - 2*a·b for unit vectors
        dot_product = tf.matmul(embeddings, embeddings, transpose_b=True)
        squared_distances = 2.0 - 2.0 * dot_product
        
        # Get positive and negative masks
        labels = tf.cast(y_true, tf.int32)
        labels_equal = tf.equal(
            tf.expand_dims(labels, 0), 
            tf.expand_dims(labels, 1)
        )
        
        # Mask out diagonal (same image)
        batch_size = tf.shape(embeddings)[0]
        diagonal_mask = tf.eye(batch_size, dtype=tf.bool)
        positive_mask = tf.logical_and(labels_equal, tf.logical_not(diagonal_mask))
        negative_mask = tf.logical_not(labels_equal)
        
        # Hardest positive: maximum distance to same-class sample
        positive_distances = tf.where(
            positive_mask,
            squared_distances,
            tf.zeros_like(squared_distances)
        )
        hardest_positive_dist = tf.reduce_max(positive_distances, axis=1)
        
        # Semi-hard negative: closest negative that's still harder than positive
        margin_mask = tf.greater(
            squared_distances,
            tf.expand_dims(hardest_positive_dist, 1)
        )
        semihard_mask = tf.logical_and(negative_mask, margin_mask)
        
        semihard_distances = tf.where(
            semihard_mask,
            squared_distances,
            tf.ones_like(squared_distances) * 1e10
        )
        hardest_negative_dist = tf.reduce_min(semihard_distances, axis=1)
        
        # Fallback to hardest negative if no semi-hard found
        has_semihard = tf.reduce_any(semihard_mask, axis=1)
        negative_distances = tf.where(
            negative_mask,
            squared_distances,
            tf.ones_like(squared_distances) * 1e10
        )
        hardest_negative_dist_fallback = tf.reduce_min(negative_distances, axis=1)
        
        hardest_negative_dist = tf.where(
            has_semihard,
            hardest_negative_dist,
            hardest_negative_dist_fallback
        )
        
        # Triplet loss
        triplet_loss = tf.maximum(
            hardest_positive_dist - hardest_negative_dist + self.margin,
            0.0
        )
        
        return tf.reduce_mean(triplet_loss)
    
    def get_config(self):
        """Get config for serialization"""
        config = super().get_config()
        config.update({
            "margin": self.margin
        })
        return config

def load_images_from_directory(data_dir):
    """Load all images and labels from directory structure"""
    images = []
    labels = []
    
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    for person_name in os.listdir(data_dir):
        person_dir = data_dir / person_name
        if not person_dir.is_dir():
            continue
        
        for image_name in os.listdir(person_dir):
            image_path = person_dir / image_name
            
            try:
                image = Image.open(image_path).convert("RGB")
                if image.size != (IMAGE_SIZE, IMAGE_SIZE):
                    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
                
                image_array = np.array(image, dtype=np.uint8)
                
                if image_array.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
                    continue
                
                images.append(image_array)
                labels.append(person_name)
            except Exception as e:
                print(f"Warning: Could not process {image_path}: {e}")
                continue
    
    return np.array(images), np.array(labels)


def train_embedding_model(
    data_dir=None,
    epochs=150,
    learning_rate=0.0001,
    margin=0.3,
    batch_size=32,
    persons_per_batch=8,
    images_per_person=4,
):
    """
    Train embedding model with triplet loss.
    
    Args:
        data_dir: Path to data directory
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        margin: Triplet loss margin
        batch_size: Batch size (should be divisible by persons_per_batch * images_per_person)
        persons_per_batch: Number of persons per batch
        images_per_person: Number of images per person per batch
    """
    from sklearn.model_selection import train_test_split
    
    CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
    
    if data_dir is None:
        data_dir = DATA_DIR
    
    print("="*60)
    print("Training CNN Embedding Model with Triplet Loss")
    print("="*60)
    
    # Load data
    print(f"\nLoading images from {data_dir}...")
    images, labels = load_images_from_directory(data_dir)
    
    if len(images) == 0:
        raise ValueError(f"No images found in {data_dir}")
    
    unique_persons = np.unique(labels)
    print(f"Loaded {len(images)} images from {len(unique_persons)} persons")
    print(f"Persons: {list(unique_persons)}")
    
    # Normalize images
    images = images.astype(np.float32) / 255.0
    
    # Split into train and validation
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(train_images)}, Validation samples: {len(val_images)}")
    
    # Adjust persons_per_batch based on available data
    num_train_persons = len(np.unique(train_labels))
    num_val_persons = len(np.unique(val_labels))
    
    # Use the minimum of requested and available persons
    actual_persons_per_batch = min(persons_per_batch, num_train_persons)
    
    if actual_persons_per_batch < persons_per_batch:
        print(f"\n⚠️  Adjusted persons_per_batch from {persons_per_batch} to {actual_persons_per_batch}")
        print(f"   (Only {num_train_persons} persons available in training set)")
    
    # Build model
    print("\nBuilding enhanced embedding model...")
    model = build_embedding_model()
    model.summary()
    
    # Compile model with triplet loss
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=TripletLoss(margin=margin)
    )
    
    # Create batch generators
    train_generator = TripletBatchGenerator(
        train_images,
        train_labels,
        persons_per_batch=actual_persons_per_batch,
        images_per_person=images_per_person,
    )
    
    val_generator = TripletBatchGenerator(
        val_images,
        val_labels,
        persons_per_batch=min(actual_persons_per_batch, num_val_persons),
        images_per_person=images_per_person,
    )
    
    # Calculate steps
    steps_per_epoch = max(1, len(train_images) // batch_size)
    validation_steps = max(1, len(val_images) // batch_size)
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size} ({persons_per_batch} persons × {images_per_person} images)")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Margin: {margin}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    
    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=30,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = callbacks.ModelCheckpoint(
        str(EMBEDDING_MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    
    lr_schedule = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train
    print("\nStarting training...")
    history = model.fit(
        train_generator(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_generator(),
        validation_steps=validation_steps,
        callbacks=[early_stopping, model_checkpoint, lr_schedule],
        verbose=1
    )
    
    # Save final model
    model.save(str(EMBEDDING_MODEL_PATH))
    print(f"\nModel saved to {EMBEDDING_MODEL_PATH}")
    
    # Print final metrics
    final_train_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]
    
    print("\nTraining completed:")
    print(f"  Final training loss: {final_train_loss:.4f}")
    print(f"  Final validation loss: {final_val_loss:.4f}")
    
    return model, history

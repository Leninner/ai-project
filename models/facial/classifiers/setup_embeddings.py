#!/usr/bin/env python3
"""
Utility script to convert the existing CNN classifier to an embedding model
and generate the embedding database.

This script:
1. Converts cnn_classifier.keras to embedding_model.keras (removes softmax layer)
2. Generates embedding database from training data
3. Sets up the system to use embedding-based authentication

Usage:
    python setup_embeddings.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.facial.classifiers import cnn_embedding


def main():
    print("=" * 60)
    print("CNN Embedding Setup Script")
    print("=" * 60)
    print()
    
    print("Step 1: Converting CNN classifier to embedding model...")
    print("-" * 60)
    try:
        embedding_model = cnn_embedding.convert_cnn_classifier_to_embedding_model()
        print("✓ Conversion successful!")
        print()
    except Exception as e:
        print(f"✗ Error during conversion: {e}")
        return 1
    
    print("Step 2: Generating embedding database...")
    print("-" * 60)
    try:
        database = cnn_embedding.save_embedding_database()
        print("✓ Database generation successful!")
        print()
    except Exception as e:
        print(f"✗ Error generating database: {e}")
        return 1
    
    print("=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print()
    print("To use embedding-based authentication, set in your .env file:")
    print("  FACIAL_CLASSIFIER_TYPE=cnn_embedding")
    print()
    print("Current files:")
    print(f"  Embedding Model: {cnn_embedding.EMBEDDING_MODEL_PATH}")
    print(f"  Embedding Database: {cnn_embedding.EMBEDDING_DATABASE_PATH}")
    print()
    print(f"Similarity Threshold: {cnn_embedding.SIMILARITY_THRESHOLD}")
    print("  (adjust in cnn_embedding.py if needed)")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
import sys
import logging
from pathlib import Path
from facial_preprocessor import FacialPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    
    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        sys.exit(1)
    
    print(f"Starting preprocessing of all images in {data_dir}")
    print("This will replace all original images with preprocessed versions.")
    
    preprocessor = FacialPreprocessor()
    results = preprocessor.preprocess_all_in_place(str(data_dir))
    
    print("\n" + "="*50)
    print("Preprocessing Summary:")
    print("="*50)
    print(f"Total processed: {results['total_processed']}")
    print(f"Total failed: {results['total_failed']}")
    print("\nBy person:")
    for person, stats in results['by_person'].items():
        print(f"  {person}: {stats['processed']} processed, {stats['failed']} failed")
    print("="*50)

if __name__ == "__main__":
    main()


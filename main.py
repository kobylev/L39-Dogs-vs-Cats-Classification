"""
Main orchestration script for the Dogs vs Cats classification pipeline.
Executes: download → prepare data → train → evaluate.
"""

import os
import sys
from pathlib import Path

from src.download_data import download_and_prepare_raw_data
from src.prepare_data import prepare_train_val_split
from src.train import train_model
from src.evaluate import evaluate_model


def main():
    """Orchestrate the full pipeline."""
    print("=" * 70)
    print("DOGS VS CATS BINARY CLASSIFICATION PIPELINE")
    print("=" * 70)
    
    # Define key paths
    data_dir = Path("data")
    output_dir = Path("output")
    
    # Create necessary directories
    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Download and clean raw data
    print("\n[STEP 1] Downloading and preparing raw dataset...")
    raw_data_dir = download_and_prepare_raw_data(data_dir)
    if raw_data_dir is None:
        print("ERROR: Failed to download dataset.")
        sys.exit(1)
    
    # Step 2: Prepare train/validation split
    print("\n[STEP 2] Creating train/validation split...")
    prepare_train_val_split(raw_data_dir, data_dir)
    
    # Step 3: Train the model
    print("\n[STEP 3] Training the CNN model...")
    model, metrics = train_model(data_dir, output_dir)
    
    # Step 4: Evaluate on validation set
    print("\n[STEP 4] Evaluating the model...")
    evaluate_model(model, data_dir, output_dir)
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"✓ Model saved to: output/model.pth")
    print(f"✓ Training curves: output/training_curves.png")
    print(f"✓ Confusion matrix: output/confusion_matrix.png")
    print(f"✓ Sample predictions: output/sample_predictions.png")
    print(f"✓ Classification report: output/classification_report.txt")
    print("=" * 70)


if __name__ == "__main__":
    main()

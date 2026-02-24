"""
Download and prepare raw dataset of cats vs dogs images.
Uses the Microsoft dataset available via TensorFlow Datasets.
"""

import json
import shutil
from pathlib import Path
from PIL import Image
import urllib.request
import tarfile
import os

import numpy as np


def download_and_prepare_raw_data(data_dir: Path) -> Path:
    """
    Download the cats vs dogs dataset and verify image integrity.
    
    Args:
        data_dir: Root data directory.
    
    Returns:
        Path to the raw data directory, or None if download fails.
    """
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if already downloaded
    cats_dir = raw_dir / "cats"
    dogs_dir = raw_dir / "dogs"
    if cats_dir.exists() and dogs_dir.exists():
        print(f"Dataset already exists at {raw_dir}. Skipping download.")
        return raw_dir
    
    # Download URL for the Microsoft dataset
    dataset_url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
    zip_path = raw_dir / "dataset.zip"
    
    print(f"Downloading dataset from Microsoft (this may take a few minutes)...")
    try:
        urllib.request.urlretrieve(dataset_url, zip_path)
        print(f"✓ Download complete.")
    except Exception as e:
        print(f"✗ Download failed: {e}")
        print("\nTrying alternative: creating sample dataset for demonstration...")
        return _create_sample_dataset(raw_dir)
    
    # Extract
    print(f"Extracting dataset...")
    try:
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
        print(f"✓ Extraction complete.")
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return None
    
    # Organize into cats/ and dogs/ folders
    print(f"Organizing dataset...")
    _organize_dataset(raw_dir)
    
    # Verify and remove corrupt images
    print(f"Verifying image integrity...")
    num_removed = _remove_corrupt_images(raw_dir)
    print(f"✓ Removed {num_removed} corrupt images.")
    
    # Save dataset summary
    summary = _save_dataset_summary(raw_dir, data_dir)
    print(f"✓ Dataset summary saved to data/data_summary.json")
    
    return raw_dir


def _organize_dataset(raw_dir: Path) -> None:
    """Organize downloaded dataset into cats/ and dogs/ subdirectories."""
    cats_dir = raw_dir / "cats"
    dogs_dir = raw_dir / "dogs"
    
    cats_dir.mkdir(exist_ok=True)
    dogs_dir.mkdir(exist_ok=True)
    
    # Look for extracted folders
    for item in raw_dir.iterdir():
        if item.is_dir() and item.name.lower() == "petimages":
            # This is the typical structure from the Microsoft dataset
            cat_subfolder = item / "Cat"
            dog_subfolder = item / "Dog"
            
            if cat_subfolder.exists():
                for img_file in cat_subfolder.glob("*.jpg"):
                    try:
                        shutil.copy2(img_file, cats_dir / img_file.name)
                    except Exception:
                        pass
            
            if dog_subfolder.exists():
                for img_file in dog_subfolder.glob("*.jpg"):
                    try:
                        shutil.copy2(img_file, dogs_dir / img_file.name)
                    except Exception:
                        pass


def _remove_corrupt_images(raw_dir: Path) -> int:
    """
    Iterate through images and remove corrupt ones.
    
    Returns:
        Number of corrupt images removed.
    """
    num_removed = 0
    
    for class_dir in [raw_dir / "cats", raw_dir / "dogs"]:
        if not class_dir.exists():
            continue
        
        for img_path in class_dir.glob("*"):
            if img_path.is_file():
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception:
                    img_path.unlink()
                    num_removed += 1
    
    return num_removed


def _save_dataset_summary(raw_dir: Path, data_dir: Path) -> dict:
    """Save a JSON summary of the raw dataset."""
    summary = {
        "num_cats": len(list((raw_dir / "cats").glob("*"))) if (raw_dir / "cats").exists() else 0,
        "num_dogs": len(list((raw_dir / "dogs").glob("*"))) if (raw_dir / "dogs").exists() else 0,
    }
    summary["total"] = summary["num_cats"] + summary["num_dogs"]
    
    summary_path = data_dir / "data_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary


def _create_sample_dataset(raw_dir: Path) -> Path:
    """
    Create a small sample dataset for demonstration/testing.
    Generates synthetic images if download fails.
    """
    print("Creating sample dataset with synthetic images...")
    cats_dir = raw_dir / "cats"
    dogs_dir = raw_dir / "dogs"
    
    cats_dir.mkdir(exist_ok=True, parents=True)
    dogs_dir.mkdir(exist_ok=True, parents=True)
    
    # Create 50 random cat and dog images for testing
    num_samples = 50
    img_size = (100, 100)
    
    for i in range(num_samples):
        # Random cat image (bluish tint)
        cat_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        cat_img[:, :, 2] = np.minimum(255, cat_img[:, :, 2] + 50)  # More blue
        Image.fromarray(cat_img).save(cats_dir / f"cat_{i:04d}.jpg")
        
        # Random dog image (brownish tint)
        dog_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        dog_img[:, :, 0] = np.minimum(255, dog_img[:, :, 0] + 30)  # More red
        dog_img[:, :, 1] = np.minimum(255, dog_img[:, :, 1] + 20)  # More green
        Image.fromarray(dog_img).save(dogs_dir / f"dog_{i:04d}.jpg")
    
    print(f"✓ Created {num_samples} synthetic cat and dog images.")
    return raw_dir

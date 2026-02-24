"""
Prepare train/validation split from raw dataset.
Randomly selects a balanced subset and splits into train/val directories.
"""

import json
import shutil
from pathlib import Path
import random


def prepare_train_val_split(
    raw_dir: Path, 
    data_dir: Path,
    num_samples_per_class: int = 1500,
    train_ratio: float = 0.67,
    seed: int = 42
) -> None:
    """
    Create balanced train/validation split from raw data.
    
    Args:
        raw_dir: Path to raw dataset with cats/ and dogs/ subdirectories.
        data_dir: Root data directory.
        num_samples_per_class: Number of images per class to use.
        train_ratio: Fraction of data for training (rest goes to validation).
        seed: Random seed for reproducibility.
    """
    random.seed(seed)
    
    # Create directory structure
    train_cats_dir = data_dir / "train" / "cats"
    train_dogs_dir = data_dir / "train" / "dogs"
    val_cats_dir = data_dir / "val" / "cats"
    val_dogs_dir = data_dir / "val" / "dogs"
    
    for d in [train_cats_dir, train_dogs_dir, val_cats_dir, val_dogs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    all_cats = list((raw_dir / "cats").glob("*"))
    all_dogs = list((raw_dir / "dogs").glob("*"))
    
    print(f"Found {len(all_cats)} cat images and {len(all_dogs)} dog images in raw data.")
    
    # Randomly select balanced subset
    selected_cats = random.sample(all_cats, min(num_samples_per_class, len(all_cats)))
    selected_dogs = random.sample(all_dogs, min(num_samples_per_class, len(all_dogs)))
    
    print(f"Selected {len(selected_cats)} cats and {len(selected_dogs)} dogs.")
    
    # Split into train/val
    num_train_cats = int(len(selected_cats) * train_ratio)
    num_train_dogs = int(len(selected_dogs) * train_ratio)
    
    train_cats = selected_cats[:num_train_cats]
    val_cats = selected_cats[num_train_cats:]
    train_dogs = selected_dogs[:num_train_dogs]
    val_dogs = selected_dogs[num_train_dogs:]
    
    # Copy files
    print("Copying files to train/val directories...")
    for img in train_cats:
        try:
            shutil.copy2(img, train_cats_dir / img.name)
        except Exception:
            pass
    
    for img in train_dogs:
        try:
            shutil.copy2(img, train_dogs_dir / img.name)
        except Exception:
            pass
    
    for img in val_cats:
        try:
            shutil.copy2(img, val_cats_dir / img.name)
        except Exception:
            pass
    
    for img in val_dogs:
        try:
            shutil.copy2(img, val_dogs_dir / img.name)
        except Exception:
            pass
    
    # Save split summary
    split_summary = {
        "train": {
            "cats": len(list(train_cats_dir.glob("*"))),
            "dogs": len(list(train_dogs_dir.glob("*")))
        },
        "val": {
            "cats": len(list(val_cats_dir.glob("*"))),
            "dogs": len(list(val_dogs_dir.glob("*")))
        }
    }
    
    split_summary["train"]["total"] = split_summary["train"]["cats"] + split_summary["train"]["dogs"]
    split_summary["val"]["total"] = split_summary["val"]["cats"] + split_summary["val"]["dogs"]
    
    summary_path = data_dir / "data_summary.json"
    with open(summary_path, "w") as f:
        json.dump(split_summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("TRAIN/VALIDATION SPLIT SUMMARY")
    print("=" * 60)
    print(f"Training set:")
    print(f"  - Cats: {split_summary['train']['cats']}")
    print(f"  - Dogs: {split_summary['train']['dogs']}")
    print(f"  - Total: {split_summary['train']['total']}")
    print(f"\nValidation set:")
    print(f"  - Cats: {split_summary['val']['cats']}")
    print(f"  - Dogs: {split_summary['val']['dogs']}")
    print(f"  - Total: {split_summary['val']['total']}")
    print("=" * 60)

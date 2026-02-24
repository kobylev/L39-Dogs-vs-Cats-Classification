"""
Evaluate the trained model on the validation set.
Generates confusion matrix, sample predictions, and classification report.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from PIL import Image

from src.model import CatDogCNN


def evaluate_model(
    model: CatDogCNN,
    data_dir: Path,
    output_dir: Path,
    batch_size: int = 32,
    device: str = None
) -> dict:
    """
    Evaluate the trained model on validation set.
    Generates confusion matrix, sample predictions, and classification report.
    
    Args:
        model: Trained model instance.
        data_dir: Path to data directory with val subdirectory.
        output_dir: Path to save evaluation plots and reports.
        batch_size: Batch size for evaluation.
        device: Device to use. Auto-detected if None.
    
    Returns:
        Dictionary with evaluation metrics.
    """
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    device = torch.device(device)
    
    # Define transforms (same as training)
    val_transforms = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Load validation dataset
    val_dataset = datasets.ImageFolder(
        str(data_dir / "val"),
        transform=val_transforms
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    class_names = val_dataset.classes  # ['cats', 'dogs']
    print(f"Classes: {class_names}")
    
    # Get predictions
    model.eval()
    all_preds = []
    all_labels = []
    all_images = []
    all_probs = []
    
    print("\nGenerating predictions on validation set...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu()
            preds = (probs > 0.5).long().squeeze()
            
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.cpu().numpy())
            all_images.append(images.cpu())
            all_probs.extend(probs.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs).flatten()
    all_images = torch.cat(all_images, dim=0)
    
    # Overall accuracy
    accuracy = (all_preds == all_labels).mean()
    print(f"\n{'=' * 60}")
    print(f"OVERALL VALIDATION ACCURACY: {accuracy:.4f}")
    print(f"{'=' * 60}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    
    # Save classification report
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(report)
        f.write("\n\n" + "=" * 70 + "\n")
        f.write(f"Overall Validation Accuracy: {accuracy:.4f}\n")
        f.write("=" * 70 + "\n")
    
    print(f"✓ Classification report saved to {report_path}")
    print("\nClassification Report:")
    print(report)
    
    # Save confusion matrix plot
    _save_confusion_matrix_plot(cm, class_names, output_dir)
    
    # Save sample predictions grid
    _save_sample_predictions_grid(all_images, all_labels, all_preds, all_probs, 
                                  val_dataset, class_names, output_dir)
    
    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    return metrics


def _save_confusion_matrix_plot(cm, class_names, output_dir):
    """Save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Normalize by row for better visualization
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax, annot_kws={'fontsize': 14})
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - Validation Set', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to {cm_path}")


def _save_sample_predictions_grid(images, labels, preds, probs, dataset, 
                                  class_names, output_dir, num_samples=16):
    """
    Save a grid of sample predictions with ground truth labels.
    Color-code titles: green for correct, red for incorrect.
    """
    # Denormalization (inverse of ImageNet normalization)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    num_samples = min(num_samples, len(images))
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Denormalize image
        img = images[i].numpy().transpose(1, 2, 0)
        img = (img * std + mean).clip(0, 1)
        
        ax.imshow(img)
        ax.axis('off')
        
        # Prepare title
        true_label = class_names[labels[i]]
        pred_label = class_names[preds[i]]
        prob = probs[i]
        
        is_correct = labels[i] == preds[i]
        title_color = 'green' if is_correct else 'red'
        
        title = f"Pred: {pred_label} ({prob:.2f})\nTrue: {true_label}"
        ax.set_title(title, fontsize=10, fontweight='bold', color=title_color)
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    samples_path = output_dir / "sample_predictions.png"
    plt.savefig(samples_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Sample predictions grid saved to {samples_path}")

"""
Training loop for the CNN model.
Trains on the prepared dataset with metrics tracking.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.model import CatDogCNN


def train_model(
    data_dir: Path,
    output_dir: Path,
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = None
) -> tuple:
    """
    Train the CNN model on the cats vs dogs dataset.
    
    Args:
        data_dir: Path to data directory with train/val subdirectories.
        output_dir: Path to save model and training curves.
        num_epochs: Number of training epochs.
        batch_size: Batch size for training and validation.
        learning_rate: Learning rate for Adam optimizer.
        device: Device to use ('cuda', 'mps', or 'cpu'). Auto-detected if None.
    
    Returns:
        Tuple of (model, metrics_dict).
    """
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Define transforms
    train_transforms = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(
        str(data_dir / "train"),
        transform=train_transforms
    )
    val_dataset = datasets.ImageFolder(
        str(data_dir / "val"),
        transform=val_transforms
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    
    # Initialize model
    model = CatDogCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print("\nStarting training...")
    print("-" * 70)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)", leave=False):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            train_correct += (preds == labels.long()).sum().item()
            train_total += labels.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)", leave=False):
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1).float()
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).long()
                val_correct += (preds == labels.long()).sum().item()
                val_total += labels.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} — Train: loss={train_loss:.4f} acc={train_acc:.4f} | Val: loss={val_loss:.4f} acc={val_acc:.4f}")
    
    print("-" * 70)
    print("Training complete!")
    
    # Save model
    model_path = output_dir / "model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✓ Model saved to {model_path}")
    
    # Save training curves
    _save_training_curves(train_losses, val_losses, train_accs, val_accs, output_dir)
    
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    
    return model, metrics


def _save_training_curves(train_losses, val_losses, train_accs, val_accs, output_dir):
    """Save training and validation curves to file."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    curves_path = output_dir / "training_curves.png"
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curves saved to {curves_path}")

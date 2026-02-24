"""
Generate Data Pipeline diagram for the README.
Creates a detailed visual representation of the data processing workflow.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(18, 10))
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(9, 9.5, 'Complete Data Pipeline: From Download to Training', 
        fontsize=22, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#2C3E50', 
                 edgecolor='black', linewidth=2.5, alpha=0.9),
        color='white')

# Colors for stages
color_download = '#FF6B6B'
color_validate = '#4ECDC4'
color_prepare = '#45B7D1'
color_split = '#FFA07A'
color_transform = '#98D8C8'
color_output = '#F7DC6F'

def draw_stage_box(ax, x, y, width, height, title, items, color):
    """Draw a stage box with items inside"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.15", 
                         edgecolor='black', facecolor=color,
                         linewidth=2.5, alpha=0.85)
    ax.add_patch(box)
    
    # Title
    ax.text(x, y + height/2 - 0.25, title, fontsize=12, 
            fontweight='bold', ha='center', va='top')
    
    # Items
    item_y = y + height/2 - 0.55
    for item in items:
        ax.text(x, item_y, f"• {item}", fontsize=9, ha='center', va='top')
        item_y -= 0.35

def draw_arrow_with_label(ax, x1, y1, x2, y2, label):
    """Draw arrow with label"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=35, 
                           linewidth=3, color='#333333')
    ax.add_patch(arrow)
    
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    ax.text(mid_x, mid_y + 0.3, label, fontsize=9, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                    edgecolor='black', linewidth=1.5, alpha=0.95),
           ha='center')

# STAGE 1: DOWNLOAD
draw_stage_box(ax, 2, 6.5, 2.8, 2.5, 'STAGE 1', 
               ['Download from', 'Microsoft Repos', '25,000+ images',
                'JPEG format'],
               color_download)

# STAGE 2: VALIDATE
draw_stage_box(ax, 5.5, 6.5, 2.8, 2.5, 'STAGE 2',
               ['Check image', 'integrity with PIL', 'Remove corrupt', 
                'files'],
               color_validate)

draw_arrow_with_label(ax, 3.4, 6.5, 4.6, 6.5, 'Raw\nImages')

# STAGE 3: PREPARE
draw_stage_box(ax, 9, 6.5, 2.8, 2.5, 'STAGE 3',
               ['Random select', '1500 cats', '1500 dogs',
                'Fixed seed'],
               color_prepare)

draw_arrow_with_label(ax, 6.4, 6.5, 7.6, 6.5, 'Valid\nImages')

# STAGE 4: SPLIT
draw_stage_box(ax, 12.5, 6.5, 2.8, 2.5, 'STAGE 4',
               ['Train/Val split', '67% train (2010)',
                '33% val (990)',
                'Balanced split'],
               color_split)

draw_arrow_with_label(ax, 10.4, 6.5, 11.6, 6.5, 'Selected\nData')

# STAGE 5: TRANSFORM
draw_stage_box(ax, 16, 6.5, 2.8, 2.5, 'STAGE 5',
               ['Resize to 150×150',
                'Normalize (ImageNet)',
                'Create batches',
                'Shuffle data'],
               color_transform)

draw_arrow_with_label(ax, 13.9, 6.5, 14.6, 6.5, 'Organized\nData')

# Output directory structure
output_y = 2.8

ax.text(9, 3.8, 'Directory Structure & Outputs', fontsize=14, fontweight='bold',
       ha='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F4F8',
                             edgecolor='black', linewidth=2))

# Left side - data/ directory
data_box = FancyBboxPatch((0.5, 0.2), 4, 3.2, boxstyle="round,pad=0.15",
                          edgecolor='black', facecolor='#F0F8FF',
                          linewidth=2, linestyle='--', alpha=0.7)
ax.add_patch(data_box)

ax.text(2.5, 3.1, 'data/', fontsize=11, fontweight='bold', ha='center',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#E0E0E0'))

data_structure = [
    'raw/',
    '├── cats/ (1500)',
    '├── dogs/ (1500)',
    'train/',
    '├── cats/ (1005)',
    '├── dogs/ (1005)',
    'val/',
    '├── cats/ (495)',
    '├── dogs/ (495)',
    'data_summary.json'
]

data_y = 2.8
for line in data_structure:
    ax.text(1, data_y, line, fontsize=8, family='monospace',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    data_y -= 0.3

# Right side - output/ directory
output_box = FancyBboxPatch((5.5, 0.2), 4, 3.2, boxstyle="round,pad=0.15",
                           edgecolor='black', facecolor='#FFF8F0',
                           linewidth=2, linestyle='--', alpha=0.7)
ax.add_patch(output_box)

ax.text(7.5, 3.1, 'output/', fontsize=11, fontweight='bold', ha='center',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#E0E0E0'))

output_structure = [
    'training_curves.png',
    '  Loss & Accuracy plots',
    'confusion_matrix.png',
    '  Prediction breakdown',
    'sample_predictions.png',
    '  16 validation examples',
    'classification_report.txt',
    '  Metrics per class',
    'model.pth',
    '  Trained weights (160MB)'
]

out_y = 2.8
for line in output_structure:
    ax.text(6, out_y, line, fontsize=8, family='monospace',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    out_y -= 0.3

# Data flow statistics box
stats_box = FancyBboxPatch((10.5, 0.2), 7, 3.2, boxstyle="round,pad=0.15",
                          edgecolor='black', facecolor='#F0FFF0',
                          linewidth=2, linestyle='--', alpha=0.7)
ax.add_patch(stats_box)

ax.text(14, 3.1, 'Pipeline Statistics', fontsize=11, fontweight='bold', ha='center',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#E0E0E0'))

stats_text = """Total Images Selected: 3,000
├── Cats: 1,500
└── Dogs: 1,500

Training Set: 2,010 images (67%)
├── Cats: 1,005
└── Dogs: 1,005

Validation Set: 990 images (33%)
├── Cats: 495
└── Dogs: 495

Image Preprocessing:
✓ Resize: 150×150 pixels
✓ Normalize: ImageNet statistics
✓ Color: RGB (3 channels)
✓ Batch Size: 32
"""

ax.text(11, 2.9, stats_text, fontsize=8, family='monospace',
       va='top', bbox=dict(boxstyle='round,pad=0.3', 
                          facecolor='white', alpha=0.9))

# Add arrows from stages to directories
draw_arrow_with_label(ax, 16, 5, 2.5, 3.4, 'Train Data')
draw_arrow_with_label(ax, 16, 5, 7.5, 3.4, 'Output\nMetrics')

plt.tight_layout()
plt.savefig('data_pipeline.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("✓ Data Pipeline diagram saved as 'data_pipeline.png'")
plt.close()

print("\nData Pipeline Diagram Features:")
print("  • 5-stage pipeline visualization")
print("  • Color-coded processing stages")
print("  • Data flow with volume information")
print("  • Directory structure visualization")
print("  • Pipeline statistics")
print("  • Output files listing")
print("  • 300 DPI high resolution")

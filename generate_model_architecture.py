"""
Generate Model Architecture diagram for the README.
Creates a visual representation of the CNN layers and data flow.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

# Title
ax.text(5, 13.5, 'CatDogCNN Architecture - Layer-by-Layer Visualization', 
        fontsize=20, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#2C3E50', 
                 edgecolor='black', linewidth=2.5, alpha=0.9),
        color='white')

# Colors for different layer types
color_input = '#E8F4F8'
color_conv = '#FF6B6B'
color_bn = '#4ECDC4'
color_pool = '#FFD93D'
color_fc = '#A8E6CF'
color_output = '#FF8C94'

y_pos = 12.5
layer_height = 0.8
layer_width = 2.5

# Helper function to draw a layer box
def draw_layer(ax, x, y, width, height, label, details, color, font_size=10):
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.1", 
                         edgecolor='black', facecolor=color,
                         linewidth=2.5, alpha=0.85)
    ax.add_patch(box)
    ax.text(x, y + height/2 - 0.15, label, fontsize=font_size, 
            fontweight='bold', ha='center', va='top')
    ax.text(x, y - height/2 + 0.15, details, fontsize=font_size-1, 
            ha='center', va='bottom')

# Helper function to draw arrow
def draw_arrow(ax, x1, y1, x2, y2, label=''):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=30, 
                           linewidth=2.5, color='#333333')
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=8, 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# INPUT LAYER
y_pos = 12
draw_layer(ax, 5, y_pos, 2.5, 0.7, 'INPUT', 
          'RGB (3×150×150)', color_input, font_size=11)
input_y = y_pos

# CONV BLOCK 1
y_pos -= 1.2

# Conv1
draw_layer(ax, 1.5, y_pos, 2.2, 0.7, 'Conv2d', 
          '3→32, 3×3, pad=1', color_conv, font_size=10)
draw_arrow(ax, 5, input_y - 0.35, 2.5, y_pos + 0.35)

# BatchNorm1
draw_layer(ax, 3.5, y_pos, 1.8, 0.7, 'BatchNorm2d', 
          '32 channels', color_bn, font_size=10)
draw_arrow(ax, 2.6, y_pos, 2.8, y_pos)

# ReLU1
draw_layer(ax, 5, y_pos, 1.8, 0.7, 'ReLU', 
          'Non-linearity', '#FFE5B4', font_size=10)
draw_arrow(ax, 4.4, y_pos, 4.1, y_pos)

# MaxPool1
draw_layer(ax, 6.5, y_pos, 1.8, 0.7, 'MaxPool2d', 
          '2×2 stride', color_pool, font_size=10)
draw_arrow(ax, 5.9, y_pos, 5.6, y_pos)

ax.text(0.5, y_pos, 'Block 1', fontsize=10, fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE5CC'))
ax.text(8, y_pos, '(32, 75, 75)', fontsize=9, style='italic',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F8F5'))

# CONV BLOCK 2
y_pos -= 1.2

draw_layer(ax, 1.5, y_pos, 2.2, 0.7, 'Conv2d', 
          '32→64, 3×3, pad=1', color_conv, font_size=10)
draw_arrow(ax, 6.5, y_pos + 1.2 - 0.35, 2.5, y_pos + 0.35)

draw_layer(ax, 3.5, y_pos, 1.8, 0.7, 'BatchNorm2d', 
          '64 channels', color_bn, font_size=10)
draw_arrow(ax, 2.6, y_pos + 1.2 - 0.35, 2.8, y_pos)

draw_layer(ax, 5, y_pos, 1.8, 0.7, 'ReLU', 
          'Non-linearity', '#FFE5B4', font_size=10)
draw_arrow(ax, 4.4, y_pos, 4.1, y_pos)

draw_layer(ax, 6.5, y_pos, 1.8, 0.7, 'MaxPool2d', 
          '2×2 stride', color_pool, font_size=10)
draw_arrow(ax, 5.9, y_pos, 5.6, y_pos)

ax.text(0.5, y_pos, 'Block 2', fontsize=10, fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE5CC'))
ax.text(8, y_pos, '(64, 37, 37)', fontsize=9, style='italic',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F8F5'))

# CONV BLOCK 3
y_pos -= 1.2

draw_layer(ax, 1.5, y_pos, 2.2, 0.7, 'Conv2d', 
          '64→128, 3×3, pad=1', color_conv, font_size=10)
draw_arrow(ax, 6.5, y_pos + 1.2 - 0.35, 2.5, y_pos + 0.35)

draw_layer(ax, 3.5, y_pos, 1.8, 0.7, 'BatchNorm2d', 
          '128 channels', color_bn, font_size=10)
draw_arrow(ax, 2.6, y_pos + 1.2 - 0.35, 2.8, y_pos)

draw_layer(ax, 5, y_pos, 1.8, 0.7, 'ReLU', 
          'Non-linearity', '#FFE5B4', font_size=10)
draw_arrow(ax, 4.4, y_pos, 4.1, y_pos)

draw_layer(ax, 6.5, y_pos, 1.8, 0.7, 'MaxPool2d', 
          '2×2 stride', color_pool, font_size=10)
draw_arrow(ax, 5.9, y_pos, 5.6, y_pos)

ax.text(0.5, y_pos, 'Block 3', fontsize=10, fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE5CC'))
ax.text(8, y_pos, '(128, 18, 18)', fontsize=9, style='italic',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F8F5'))

# FLATTEN
y_pos -= 1.2
draw_layer(ax, 5, y_pos, 2.5, 0.7, 'Flatten', 
          '128×18×18 → 41,472', '#D4A5FF', font_size=10)
draw_arrow(ax, 6.5, y_pos + 1.2 - 0.35, 5, y_pos + 0.35)

# FULLY CONNECTED LAYERS
y_pos -= 1.2

draw_layer(ax, 2, y_pos, 2.2, 0.7, 'Linear', 
          '41,472 → 256', color_fc, font_size=10)
draw_arrow(ax, 5, y_pos + 1.2 - 0.35, 3, y_pos + 0.35)

draw_layer(ax, 4.5, y_pos, 1.8, 0.7, 'ReLU', 
          'Non-linearity', '#FFE5B4', font_size=10)
draw_arrow(ax, 3.1, y_pos, 3.6, y_pos)

draw_layer(ax, 6.5, y_pos, 1.8, 0.7, 'Dropout', 
          'p=0.5 (train)', '#C1FFD7', font_size=10)
draw_arrow(ax, 5.4, y_pos, 5.6, y_pos)

ax.text(0.5, y_pos, 'FC Layer 1', fontsize=10, fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE5CC'))

# OUTPUT LAYER
y_pos -= 1.2

draw_layer(ax, 5, y_pos, 2.5, 0.7, 'Linear', 
          '256 → 1 (Binary Logit)', color_output, font_size=11)
draw_arrow(ax, 6.5, y_pos + 1.2 - 0.35, 5, y_pos + 0.35)

ax.text(0.5, y_pos, 'Output', fontsize=10, fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE5CC'))

# PREDICTIONS
y_pos -= 1.2

ax.text(2.5, y_pos, 'Sigmoid(z)', fontsize=11, fontweight='bold', ha='center',
       bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F4F8', 
                edgecolor='black', linewidth=2))
ax.text(5, y_pos, 'Threshold @ 0.5', fontsize=11, fontweight='bold', ha='center',
       bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F4F8',
                edgecolor='black', linewidth=2))
ax.text(7.5, y_pos, 'Cat or Dog', fontsize=11, fontweight='bold', ha='center',
       bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F4F8',
                edgecolor='black', linewidth=2))

draw_arrow(ax, 3.75, y_pos, 3.15, y_pos)
draw_arrow(ax, 5.75, y_pos, 5.3, y_pos)

# LEGEND
legend_y = 0.7
ax.text(0.2, legend_y, 'Layer Types:', fontsize=10, fontweight='bold')

items = [
    ('Conv Layers', color_conv),
    ('Batch Norm', color_bn),
    ('Pooling', color_pool),
    ('FC Layers', color_fc),
    ('Output', color_output)
]

legend_x_start = 0.2
for i, (label, color) in enumerate(items):
    x = legend_x_start + (i % 5) * 1.9
    y = legend_y - 0.5 - (i // 5) * 0.4
    rect = mpatches.Rectangle((x, y), 0.3, 0.2, 
                              facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + 0.4, y + 0.1, label, fontsize=9, va='center')

# Add statistics
stats_text = (
    "Network Statistics:\n"
    "• Total Parameters: ~10.7M\n"
    "• Trainable Params: All\n"
    "• Input: RGB images (3 channels)\n"
    "• Output: Binary classification (Cat=0, Dog=1)\n"
    "• Loss Function: BCEWithLogitsLoss"
)

ax.text(8.5, 0.3, stats_text, fontsize=9, 
       bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9E6',
                edgecolor='black', linewidth=1.5),
       verticalalignment='bottom', family='monospace')

plt.tight_layout()
plt.savefig('model_architecture.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("✓ Model Architecture diagram saved as 'model_architecture.png'")
plt.close()

print("\nModel Architecture Diagram Features:")
print("  • 3 convolutional blocks with progressive channel expansion")
print("  • BatchNorm and ReLU after each convolution")
print("  • MaxPooling layers for dimensionality reduction")
print("  • 2 fully connected layers")
print("  • Dropout for regularization")
print("  • Binary output with sigmoid prediction")
print("  • 300 DPI high resolution")

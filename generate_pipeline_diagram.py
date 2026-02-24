"""
Generate a visual pipeline diagram for the README.
Creates a colorful flowchart showing the data pipeline steps.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Color scheme (different from typical)
color_download = '#FF6B6B'      # Red/coral
color_validate = '#4ECDC4'      # Teal
color_prepare = '#45B7D1'       # Blue
color_train = '#FFA07A'         # Light salmon
color_evaluate = '#98D8C8'      # Mint
color_output = '#F7DC6F'        # Yellow/gold

# Title
ax.text(5, 9.5, 'Dogs vs Cats Classification Pipeline', 
        fontsize=24, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#2C3E50', 
                 edgecolor='black', linewidth=2.5, alpha=0.9),
        color='white')

# Step 1: Download Data
box1 = FancyBboxPatch((0.3, 7.5), 1.8, 1.2, 
                       boxstyle="round,pad=0.1", 
                       edgecolor='black', facecolor=color_download, 
                       linewidth=2.5, alpha=0.85)
ax.add_patch(box1)
ax.text(1.2, 8.3, 'Download', fontsize=11, fontweight='bold', ha='center')
ax.text(1.2, 7.95, 'Dataset', fontsize=11, fontweight='bold', ha='center')
ax.text(1.2, 7.65, '(25K+ imgs)', fontsize=8, ha='center', style='italic')

# Arrow 1
arrow1 = FancyArrowPatch((2.15, 8.1), (3.15, 8.1),
                        arrowstyle='->', mutation_scale=25, 
                        linewidth=2.5, color='#333333')
ax.add_patch(arrow1)

# Step 2: Validate & Clean
box2 = FancyBboxPatch((3.2, 7.5), 1.8, 1.2,
                       boxstyle="round,pad=0.1",
                       edgecolor='black', facecolor=color_validate,
                       linewidth=2.5, alpha=0.85)
ax.add_patch(box2)
ax.text(4.1, 8.3, 'Validate &', fontsize=11, fontweight='bold', ha='center')
ax.text(4.1, 7.95, 'Clean Data', fontsize=11, fontweight='bold', ha='center')
ax.text(4.1, 7.65, '(remove corrupt)', fontsize=8, ha='center', style='italic')

# Arrow 2
arrow2 = FancyArrowPatch((5.05, 8.1), (6.05, 8.1),
                        arrowstyle='->', mutation_scale=25,
                        linewidth=2.5, color='#333333')
ax.add_patch(arrow2)

# Step 3: Prepare & Split
box3 = FancyBboxPatch((6.1, 7.5), 1.8, 1.2,
                       boxstyle="round,pad=0.1",
                       edgecolor='black', facecolor=color_prepare,
                       linewidth=2.5, alpha=0.85)
ax.add_patch(box3)
ax.text(7.0, 8.3, 'Prepare &', fontsize=11, fontweight='bold', ha='center')
ax.text(7.0, 7.95, 'Split Data', fontsize=11, fontweight='bold', ha='center')
ax.text(7.0, 7.65, '(train/val)', fontsize=8, ha='center', style='italic')

# Arrow 3
arrow3 = FancyArrowPatch((7.95, 8.1), (8.95, 8.1),
                        arrowstyle='->', mutation_scale=25,
                        linewidth=2.5, color='#333333')
ax.add_patch(arrow3)

# Step 4: Model Training (large box on second row)
box4 = FancyBboxPatch((0.3, 5.5), 3.5, 1.5,
                       boxstyle="round,pad=0.1",
                       edgecolor='black', facecolor=color_train,
                       linewidth=2.5, alpha=0.85)
ax.add_patch(box4)
ax.text(2.05, 6.65, 'Train CNN Model', fontsize=12, fontweight='bold', ha='center')
ax.text(2.05, 6.3, '• 3 Conv Blocks', fontsize=9, ha='center')
ax.text(2.05, 6.0, '• 20 Epochs', fontsize=9, ha='center')
ax.text(2.05, 5.7, '• Track Loss & Accuracy', fontsize=9, ha='center')

# Arrow 4 (down from box3)
arrow4 = FancyArrowPatch((7.0, 7.5), (7.0, 6.5),
                        arrowstyle='->', mutation_scale=25,
                        linewidth=2.5, color='#333333')
ax.add_patch(arrow4)

# Step 5: Parallel processing - Data side
box5a = FancyBboxPatch((8.8, 6.8), 1.0, 0.6,
                        boxstyle="round,pad=0.05",
                        edgecolor='black', facecolor='#FFE5CC',
                        linewidth=2, alpha=0.85)
ax.add_patch(box5a)
ax.text(9.3, 7.1, 'Training', fontsize=9, fontweight='bold', ha='center')
ax.text(9.3, 6.88, 'Set', fontsize=8, ha='center')

box5b = FancyBboxPatch((8.8, 5.8), 1.0, 0.6,
                        boxstyle="round,pad=0.05",
                        edgecolor='black', facecolor='#FFE5CC',
                        linewidth=2, alpha=0.85)
ax.add_patch(box5b)
ax.text(9.3, 6.1, 'Validation', fontsize=9, fontweight='bold', ha='center')
ax.text(9.3, 5.88, 'Set', fontsize=8, ha='center')

# Arrows from box4 to box5
arrow5a = FancyArrowPatch((3.8, 6.8), (8.8, 7.1),
                         arrowstyle='->', mutation_scale=20,
                         linewidth=2, color='#555555', linestyle='dashed')
ax.add_patch(arrow5a)

arrow5b = FancyArrowPatch((3.8, 5.9), (8.8, 6.1),
                         arrowstyle='->', mutation_scale=20,
                         linewidth=2, color='#555555', linestyle='dashed')
ax.add_patch(arrow5b)

# Step 6: Evaluation (large box on third row)
box6 = FancyBboxPatch((0.3, 3.3), 3.5, 1.8,
                       boxstyle="round,pad=0.1",
                       edgecolor='black', facecolor=color_evaluate,
                       linewidth=2.5, alpha=0.85)
ax.add_patch(box6)
ax.text(2.05, 4.8, 'Evaluate Model', fontsize=12, fontweight='bold', ha='center')
ax.text(2.05, 4.4, '• Confusion Matrix', fontsize=9, ha='center')
ax.text(2.05, 4.05, '• Classification Report', fontsize=9, ha='center')
ax.text(2.05, 3.7, '• Sample Predictions', fontsize=9, ha='center')

# Arrow 6
arrow6 = FancyArrowPatch((2.05, 5.5), (2.05, 5.15),
                        arrowstyle='->', mutation_scale=25,
                        linewidth=2.5, color='#333333')
ax.add_patch(arrow6)

# Step 7: Output files (three boxes)
output_y = 0.8
output_height = 1.2

box7a = FancyBboxPatch((4.5, output_y), 1.4, output_height,
                        boxstyle="round,pad=0.08",
                        edgecolor='black', facecolor=color_output,
                        linewidth=2.5, alpha=0.85)
ax.add_patch(box7a)
ax.text(5.2, 1.75, 'Training', fontsize=10, fontweight='bold', ha='center')
ax.text(5.2, 1.45, 'Curves', fontsize=10, fontweight='bold', ha='center')
ax.text(5.2, 1.1, '(Loss &', fontsize=8, ha='center')
ax.text(5.2, 0.85, 'Accuracy)', fontsize=8, ha='center')

box7b = FancyBboxPatch((6.15, output_y), 1.4, output_height,
                        boxstyle="round,pad=0.08",
                        edgecolor='black', facecolor=color_output,
                        linewidth=2.5, alpha=0.85)
ax.add_patch(box7b)
ax.text(6.85, 1.75, 'Confusion', fontsize=10, fontweight='bold', ha='center')
ax.text(6.85, 1.45, 'Matrix', fontsize=10, fontweight='bold', ha='center')
ax.text(6.85, 1.1, '& Sample', fontsize=8, ha='center')
ax.text(6.85, 0.85, 'Predictions', fontsize=8, ha='center')

box7c = FancyBboxPatch((7.8, output_y), 1.4, output_height,
                        boxstyle="round,pad=0.08",
                        edgecolor='black', facecolor=color_output,
                        linewidth=2.5, alpha=0.85)
ax.add_patch(box7c)
ax.text(8.5, 1.75, 'Model &', fontsize=10, fontweight='bold', ha='center')
ax.text(8.5, 1.45, 'Metrics', fontsize=10, fontweight='bold', ha='center')
ax.text(8.5, 1.1, '(Precision,', fontsize=8, ha='center')
ax.text(8.5, 0.85, 'Recall, F1)', fontsize=8, ha='center')

# Arrows from evaluation to outputs
arrow7a = FancyArrowPatch((2.05, 3.3), (5.2, 2.0),
                         arrowstyle='->', mutation_scale=25,
                         linewidth=2.5, color='#333333')
ax.add_patch(arrow7a)

arrow7b = FancyArrowPatch((2.05, 3.3), (6.85, 2.0),
                         arrowstyle='->', mutation_scale=25,
                         linewidth=2.5, color='#333333')
ax.add_patch(arrow7b)

arrow7c = FancyArrowPatch((2.05, 3.3), (8.5, 2.0),
                         arrowstyle='->', mutation_scale=25,
                         linewidth=2.5, color='#333333')
ax.add_patch(arrow7c)

# Final step indicator
step1_rect = FancyBboxPatch((8.8, 4.8), 1.0, 0.5,
                            boxstyle="round,pad=0.05",
                            edgecolor='black', facecolor='#E8F8F5',
                            linewidth=2, alpha=0.85)
ax.add_patch(step1_rect)
ax.text(9.3, 5.05, 'Step 1-3', fontsize=8, fontweight='bold', ha='center')

# Arrow to final step
arrow_final1 = FancyArrowPatch((7.95, 8.1), (8.8, 5.05),
                              arrowstyle='->', mutation_scale=20,
                              linewidth=2, color='#AAA', linestyle='dotted', alpha=0.6)
ax.add_patch(arrow_final1)

# Add legend/notes at bottom
legend_y = -0.3
ax.text(0.3, legend_y, '📊 Input Data', fontsize=9, fontweight='bold')
ax.text(2.5, legend_y, '🔄 Processing', fontsize=9, fontweight='bold')
ax.text(4.7, legend_y, '🤖 Training', fontsize=9, fontweight='bold')
ax.text(6.2, legend_y, '📈 Evaluation', fontsize=9, fontweight='bold')
ax.text(8.2, legend_y, '💾 Output', fontsize=9, fontweight='bold')

# Save figure
plt.tight_layout()
plt.savefig('pipeline_diagram.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✓ Pipeline diagram saved as 'pipeline_diagram.png'")
plt.close()

print("\nDiagram features:")
print("  • Colorful 5-stage pipeline visualization")
print("  • Shows data flow from download to output")
print("  • Different style and colors from reference")
print("  • High resolution (300 DPI)")
print("  • Ready for GitHub/documentation")

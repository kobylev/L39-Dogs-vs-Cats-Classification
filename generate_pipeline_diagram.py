"""
Generate a professional vertical Data Pipeline diagram.
Saves to output/data_pipeline.png.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def create_pipeline_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15)
    ax.axis('off')

    # Color scheme
    colors = {
        'source': '#AED6F1',   # Light Blue
        'process': '#D2B4DE',  # Light Purple
        'split': '#F9E79F',    # Light Yellow
        'train': '#F5B7B1',    # Light Red
        'eval': '#A9DFBF',     # Light Green
        'output': '#E5E7E9'    # Light Grey
    }

    def draw_step(x, y, w, h, title, details, color):
        box = FancyBboxPatch((x, y), w, h, 
                             boxstyle="round,pad=0.1", 
                             edgecolor='black', facecolor=color, 
                             linewidth=1.5, alpha=0.9)
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.6, title, fontsize=12, fontweight='bold', ha='center')
        ax.text(x + w/2, y + h*0.3, details, fontsize=9, ha='center', style='italic')
        return (x + w/2, y), (x + w/2, y + h)

    # Vertical sequence
    curr_y = 13.0
    
    # 1. Raw Data
    _, s1_top = draw_step(3, curr_y, 4, 1.2, "1. Raw Data Source", "Microsoft Dogs vs Cats (25K imgs)", colors['source'])
    
    # 2. Validation
    curr_y -= 2.0
    s2_bot, s2_top = draw_step(3, curr_y, 4, 1.2, "2. Data Cleaning", "Remove Corrupt & Non-Image files", colors['process'])
    
    # 3. Preparation
    curr_y -= 2.0
    s3_bot, s3_top = draw_step(3, curr_y, 4, 1.2, "3. Data Splitting", "80% Train | 20% Validation", colors['split'])
    
    # 4. Training
    curr_y -= 2.0
    s4_bot, s4_top = draw_step(3, curr_y, 4, 1.2, "4. Model Training", "CNN Learning (20 Epochs)", colors['train'])
    
    # 5. Evaluation
    curr_y -= 2.0
    s5_bot, s5_top = draw_step(3, curr_y, 4, 1.2, "5. Performance Evaluation", "Precision, Recall, F1 & Confusion Matrix", colors['eval'])
    
    # 6. Artifacts
    curr_y -= 2.0
    s6_bot, s6_top = draw_step(3, curr_y, 4, 1.2, "6. Results Export", "Plots, Reports & Saved Model", colors['output'])

    # Connections
    def arrow(start, end):
        arr = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=20, linewidth=2, color='#2C3E50')
        ax.add_patch(arr)

    arrow((5, 13.0), (5, 12.2)) # 1 -> 2
    arrow((5, 11.0), (5, 10.2)) # 2 -> 3
    arrow((5, 9.0), (5, 8.2))   # 3 -> 4
    arrow((5, 7.0), (5, 6.2))   # 4 -> 5
    arrow((5, 5.0), (5, 4.2))   # 5 -> 6

    plt.title("End-to-End Data Pipeline Flow", fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('output/data_pipeline.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Data pipeline diagram saved as 'output/data_pipeline.png'")

if __name__ == "__main__":
    create_pipeline_diagram()

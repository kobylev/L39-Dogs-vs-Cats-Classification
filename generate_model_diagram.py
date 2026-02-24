"""
Generate a detailed visualization of the CNN Architecture.
Shows the flow from Input (150x150x3) through Conv Blocks to Output.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def create_model_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')

    # Color scheme
    colors = {
        'input': '#E8F8F5',
        'conv': '#D1F2EB',
        'bn': '#A9DFBF',
        'pool': '#7DCEA0',
        'fc': '#FAD7A0',
        'dropout': '#F5B041',
        'output': '#EDBB99'
    }

    def draw_layer(x, y, w, h, title, details, color):
        box = FancyBboxPatch((x, y), w, h, 
                             boxstyle="round,pad=0.1", 
                             edgecolor='black', facecolor=color, 
                             linewidth=1.5, alpha=0.9)
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.6, title, fontsize=12, fontweight='bold', ha='center')
        ax.text(x + w/2, y + h*0.3, details, fontsize=10, ha='center')
        return (x + w/2, y), (x + w/2, y + h)

    # Layer positions (Vertical Stack)
    curr_y = 14.5
    
    # Input
    _, in_top = draw_layer(3, curr_y, 4, 1.0, "Input Image", "150 x 150 x 3 (RGB)", colors['input'])
    
    # Conv Block 1
    curr_y -= 1.8
    bot, top = draw_layer(3, curr_y, 4, 1.2, "Conv Block 1", "32 filters | 3x3 | BN | MaxPool", colors['conv'])
    
    # Conv Block 2
    curr_y -= 1.8
    bot2, top2 = draw_layer(3, curr_y, 4, 1.2, "Conv Block 2", "64 filters | 3x3 | BN | MaxPool", colors['conv'])
    
    # Conv Block 3
    curr_y -= 1.8
    bot3, top3 = draw_layer(3, curr_y, 4, 1.2, "Conv Block 3", "128 filters | 3x3 | BN | MaxPool", colors['conv'])
    
    # Flatten
    curr_y -= 1.2
    bot4, top4 = draw_layer(3, curr_y, 4, 0.8, "Flatten", "128 x 18 x 18 = 41,472", '#D5D8DC')
    
    # FC 1
    curr_y -= 1.5
    bot5, top5 = draw_layer(3, curr_y, 4, 1.0, "Fully Connected 1", "256 Units | ReLU | Dropout(0.5)", colors['fc'])
    
    # Output
    curr_y -= 1.5
    bot6, top6 = draw_layer(3, curr_y, 4, 0.8, "Output Layer", "1 Unit (Logits)", colors['output'])

    # Connections
    def arrow(start, end):
        arr = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=20, linewidth=2, color='#2C3E50')
        ax.add_patch(arr)

    arrow((5, 14.5), (5, 14.3)) # From Input
    arrow((5, 13.1), (5, 12.8)) # 1 -> 2
    arrow((5, 11.3), (5, 11.0)) # 2 -> 3
    arrow((5, 9.5), (5, 9.2))   # 3 -> 4
    arrow((5, 8.3), (5, 8.0))   # 4 -> 5
    arrow((5, 7.0), (5, 6.7))   # 5 -> 6

    plt.title("CatDogCNN Architecture Details", fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('output/cnn_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Architecture diagram saved as 'output/cnn_architecture.png'")

if __name__ == "__main__":
    create_model_diagram()

"""
Create GNN architecture visualization showing:
- Agent graph structure
- Virtual node connections
- Message passing layers
- Embeddings evolution
- Final prediction
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

def create_gnn_visualization():
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle('GNN Architecture: From Graph to Prediction', fontsize=16, fontweight='bold')
    
    # Color scheme
    node_colors = {
        'Solver': '#FF6B6B',
        'Extract': '#4ECDC4',
        'Decompose': '#45B7D1',
        'Combine': '#FFA07A',
        'Virtual': '#FFD93D'
    }
    
    # ========== Panel 1: Input Graph with Virtual Node ==========
    ax1 = axes[0]
    ax1.set_xlim(-1, 3)
    ax1.set_ylim(-1, 5)
    ax1.axis('off')
    ax1.set_title('1. Input Graph\n+ Virtual Node', fontsize=12, fontweight='bold')
    
    # Agent nodes
    agent_positions = {
        'Solver': (0.5, 4),
        'Extract': (2, 3.5),
        'Decompose': (2, 2),
        'Combine': (0.5, 1)
    }
    
    # Draw edges between agents
    edges = [
        ('Solver', 'Extract'),
        ('Extract', 'Decompose'),
        ('Decompose', 'Combine')
    ]
    
    for src, dst in edges:
        src_pos = agent_positions[src]
        dst_pos = agent_positions[dst]
        arrow = FancyArrowPatch(src_pos, dst_pos, 
                               arrowstyle='->', mutation_scale=20, 
                               linewidth=2, color='gray', zorder=1)
        ax1.add_patch(arrow)
    
    # Draw agent nodes
    for node_name, pos in agent_positions.items():
        circle = Circle(pos, 0.3, color=node_colors[node_name], 
                       ec='black', linewidth=2, zorder=3)
        ax1.add_patch(circle)
        ax1.text(pos[0], pos[1], node_name.split('_')[0][:4], 
                ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Virtual node at center
    virtual_pos = (1.25, 2.5)
    circle = Circle(virtual_pos, 0.35, color=node_colors['Virtual'], 
                   ec='red', linewidth=3, zorder=4)
    ax1.add_patch(circle)
    ax1.text(virtual_pos[0], virtual_pos[1], 'Virt', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Virtual node connections (dashed)
    for node_name, pos in agent_positions.items():
        arrow = FancyArrowPatch(virtual_pos, pos, 
                               arrowstyle='<->', mutation_scale=15, 
                               linewidth=1.5, color='red', 
                               linestyle='dashed', zorder=2)
        ax1.add_patch(arrow)
    
    ax1.text(1.25, 0, 'Virtual node\nconnects to all', 
            ha='center', fontsize=9, style='italic')
    
    # ========== Panel 2: Message Passing Layers ==========
    ax2 = axes[1]
    ax2.set_xlim(-0.5, 4)
    ax2.set_ylim(-0.5, 5)
    ax2.axis('off')
    ax2.set_title('2. Message Passing\n(4 Layers)', fontsize=12, fontweight='bold')
    
    # Show layers vertically
    layer_names = ['Input\n(One-hot)', 'Layer 1\nGNN', 'Layer 2\nGNN', 'Layer 3\nGNN', 'Layer 4\nGNN']
    layer_dims = ['15-dim', '32-dim', '32-dim', '32-dim', '32-dim']
    
    num_layers = 5
    y_positions = np.linspace(4, 0.5, num_layers)
    
    for i, (y, name, dim) in enumerate(zip(y_positions, layer_names, layer_dims)):
        # Draw nodes at this layer
        x_positions = [0.5, 1.5, 2.5, 3.5]
        colors_at_layer = list(node_colors.values())
        
        for j, (x, color) in enumerate(zip(x_positions, colors_at_layer)):
            # Gradually change intensity to show embedding evolution
            alpha = 0.3 + 0.7 * (i / num_layers)
            circle = Circle((x, y), 0.15, color=color, alpha=alpha,
                           ec='black', linewidth=1, zorder=3)
            ax2.add_patch(circle)
            
            # Draw arrows to next layer
            if i < num_layers - 1:
                next_y = y_positions[i + 1]
                arrow = FancyArrowPatch((x, y - 0.15), (x, next_y + 0.15),
                                       arrowstyle='->', mutation_scale=10,
                                       linewidth=1, color='gray', alpha=0.5)
                ax2.add_patch(arrow)
        
        # Layer label
        ax2.text(-0.3, y, name, ha='right', va='center', fontsize=8, fontweight='bold')
        ax2.text(4.2, y, dim, ha='left', va='center', fontsize=7, style='italic')
    
    # Annotation
    ax2.annotate('Embeddings\nevolve', xy=(2, 2.5), xytext=(3.8, 3.5),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                fontsize=9, color='blue', fontweight='bold')
    
    # ========== Panel 3: Virtual Node Readout ==========
    ax3 = axes[2]
    ax3.set_xlim(-1, 3)
    ax3.set_ylim(-1, 5)
    ax3.axis('off')
    ax3.set_title('3. Virtual Node\nReadout', fontsize=12, fontweight='bold')
    
    # Final layer nodes (small)
    final_positions = [(0, 4), (0.7, 4), (1.4, 4), (2.1, 4)]
    final_colors = list(node_colors.values())[:4]
    
    for pos, color in zip(final_positions, final_colors):
        circle = Circle(pos, 0.15, color=color, ec='black', linewidth=1)
        ax3.add_patch(circle)
    
    ax3.text(1.05, 4.5, 'Final layer node embeddings', ha='center', fontsize=9)
    
    # Virtual node (large)
    virtual_final_pos = (1.05, 2.5)
    circle = Circle(virtual_final_pos, 0.4, color=node_colors['Virtual'], 
                   ec='red', linewidth=3, zorder=4)
    ax3.add_patch(circle)
    ax3.text(virtual_final_pos[0], virtual_final_pos[1], 'Virtual\nNode', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows from nodes to virtual node
    for pos in final_positions:
        arrow = FancyArrowPatch(pos, (virtual_final_pos[0], virtual_final_pos[1] + 0.3),
                               arrowstyle='->', mutation_scale=15,
                               linewidth=2, color='red')
        ax3.add_patch(arrow)
    
    # Virtual node embedding representation
    emb_box = FancyBboxPatch((0.3, 1.2), 1.5, 0.6, 
                             boxstyle="round,pad=0.1", 
                             facecolor='lightyellow', 
                             edgecolor='orange', linewidth=2)
    ax3.add_patch(emb_box)
    ax3.text(1.05, 1.5, 'Graph Embedding\n[32-dim vector]', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    arrow = FancyArrowPatch((virtual_final_pos[0], virtual_final_pos[1] - 0.4), 
                           (1.05, 1.8),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2.5, color='orange')
    ax3.add_patch(arrow)
    
    ax3.text(1.05, 0.5, 'Aggregated global\ngraph information', 
            ha='center', fontsize=9, style='italic', color='green')
    
    # ========== Panel 4: MLP Prediction ==========
    ax4 = axes[3]
    ax4.set_xlim(-0.5, 2.5)
    ax4.set_ylim(-1, 5)
    ax4.axis('off')
    ax4.set_title('4. Prediction\n(MLP + Sigmoid)', fontsize=12, fontweight='bold')
    
    # Graph embedding input
    emb_box = FancyBboxPatch((0.2, 3.5), 1.6, 0.8, 
                             boxstyle="round,pad=0.1", 
                             facecolor='lightyellow', 
                             edgecolor='orange', linewidth=2)
    ax4.add_patch(emb_box)
    ax4.text(1, 3.9, 'Graph Embedding\n[32-dim]', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrow down
    arrow = FancyArrowPatch((1, 3.5), (1, 3),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2.5, color='black')
    ax4.add_patch(arrow)
    
    # MLP layer
    mlp_box = FancyBboxPatch((0.3, 2.2), 1.4, 0.6, 
                            boxstyle="round,pad=0.1", 
                            facecolor='lightblue', 
                            edgecolor='blue', linewidth=2)
    ax4.add_patch(mlp_box)
    ax4.text(1, 2.5, 'Linear(32 → 1)', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrow down
    arrow = FancyArrowPatch((1, 2.2), (1, 1.8),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2.5, color='black')
    ax4.add_patch(arrow)
    
    # Sigmoid
    sigmoid_box = FancyBboxPatch((0.3, 1.2), 1.4, 0.6, 
                                boxstyle="round,pad=0.1", 
                                facecolor='lightgreen', 
                                edgecolor='green', linewidth=2)
    ax4.add_patch(sigmoid_box)
    ax4.text(1, 1.5, 'Sigmoid(·)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrow down
    arrow = FancyArrowPatch((1, 1.2), (1, 0.8),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2.5, color='black')
    ax4.add_patch(arrow)
    
    # Final prediction
    pred_circle = Circle((1, 0.2), 0.4, color='gold', 
                        ec='darkorange', linewidth=3, zorder=4)
    ax4.add_patch(pred_circle)
    ax4.text(1, 0.2, '0.87', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    ax4.text(1, -0.5, 'Predicted\nAccuracy', 
            ha='center', fontsize=9, style='italic', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figures/gnn_architecture_detailed.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/gnn_architecture_detailed.png")
    plt.close()

def create_pipeline_diagram():
    """Create detailed pipeline diagram"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors for each step
    colors = {
        'gen': '#3498db',      # Blue
        'pred': '#2ecc71',     # Green
        'sel': '#f39c12',      # Orange
        'eval': '#e74c3c',     # Red
        'update': '#f1c40f',   # Yellow
        'retrain': '#9b59b6'   # Purple
    }
    
    # Step positions in hexagonal layout (perfect hexagon with 60° spacing)
    angles = [90, 30, -30, -90, -150, 150]  # Regular hexagon vertices
    radius = 3
    center = (5, 5)
    
    positions = []
    for angle in angles:
        rad = np.radians(angle)
        x = center[0] + radius * np.cos(rad)
        y = center[1] + radius * np.sin(rad)
        positions.append((x, y))
    
    # Step information
    steps = [
        ('1. Graph\nGeneration', colors['gen'], '200k graphs\nmax 8 nodes'),
        ('2. GNN\nPrediction', colors['pred'], 'Predict scores\nfor all'),
        ('3. Candidate\nSelection', colors['sel'], 'Top 10 → pool\nTop 5 → eval'),
        ('4. LLM\nEvaluation', colors['eval'], '5 math problems\nper graph'),
        ('5. Data\nUpdate', colors['update'], 'Add to\ntraining set'),
        ('6. GNN\nRetraining', colors['retrain'], '300 epochs\nMSE loss')
    ]
    
    # Draw steps
    for i, (pos, (title, color, desc)) in enumerate(zip(positions, steps)):
        # Box
        box = FancyBboxPatch((pos[0]-0.8, pos[1]-0.5), 1.6, 1, 
                            boxstyle="round,pad=0.1", 
                            facecolor=color, alpha=0.3,
                            edgecolor=color, linewidth=3)
        ax.add_patch(box)
        
        # Title
        ax.text(pos[0], pos[1]+0.25, title, 
               ha='center', va='center', fontsize=11, fontweight='bold')
        
        # Description
        ax.text(pos[0], pos[1]-0.15, desc, 
               ha='center', va='center', fontsize=8, style='italic')
        
        # Arrow to next step
        if i < len(positions) - 1:
            next_pos = positions[i+1]
            # Calculate arrow start and end
            angle_to_next = np.arctan2(next_pos[1] - pos[1], next_pos[0] - pos[0])
            start_x = pos[0] + 0.8 * np.cos(angle_to_next)
            start_y = pos[1] + 0.5 * np.cos(angle_to_next)
            end_x = next_pos[0] - 0.8 * np.cos(angle_to_next)
            end_y = next_pos[1] - 0.5 * np.cos(angle_to_next)
            
            arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                                   arrowstyle='->', mutation_scale=30,
                                   linewidth=3, color='black')
            ax.add_patch(arrow)
    
    # Closing arrow from step 6 to step 1
    arrow = FancyArrowPatch((positions[5][0]-0.5, positions[5][1]+0.3), 
                           (positions[0][0]-0.5, positions[0][1]+0.3),
                           arrowstyle='->', mutation_scale=30,
                           linewidth=3, color='black',
                           connectionstyle="arc3,rad=0.5")
    ax.add_patch(arrow)
    
    # Center text
    center_circle = Circle(center, 1.2, color='white', 
                          ec='black', linewidth=2, zorder=10)
    ax.add_patch(center_circle)
    ax.text(center[0], center[1]+0.3, 'GraphMind', 
           ha='center', va='center', fontsize=14, fontweight='bold', zorder=11)
    ax.text(center[0], center[1]-0.1, 'Pipeline', 
           ha='center', va='center', fontsize=12, fontweight='bold', zorder=11)
    ax.text(center[0], center[1]-0.5, '30 iterations', 
           ha='center', va='center', fontsize=9, style='italic', zorder=11)
    
    # Title
    ax.text(5, 9.5, 'GraphMind: Iterative 6-Step Pipeline', 
           ha='center', va='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figures/pipeline_diagram.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/pipeline_diagram.png")
    plt.close()

if __name__ == '__main__':
    create_gnn_visualization()
    create_pipeline_diagram()
    print("\nVisualizations created successfully!")
    print("Use these in your presentation to replace placeholder images.")


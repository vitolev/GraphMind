"""
Create a LaTeX-style table PNG for the Medium post.

This script generates a professional-looking table comparing GNN model architectures.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Set high DPI for Medium-quality images
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.facecolor'] = 'white'

def create_table_image(output_path: Path, table_type: str = 'architecture'):
    """Create a LaTeX-style table comparing GNN models."""
    
    if table_type == 'architecture':
        # Table data for architecture comparison
        headers = ['Model', 'Layers', 'Hidden Dim', 'Epochs', 'Dropout', 'Avg. Validation Loss']
        data = [
            ['GCN', '3', '16', '300', '0.0', '0.0057'],
            ['GraphSAGE', '3', '32', '300', '0.2', '0.0056'],
            ['GAT', '4', '8', '300', '0.1', '0.0058'],
        ]
        title = 'GNN Model Architectures and Validation Performance'
        figsize = (14, 3)
    else:  # test_loss
        # Table data for test loss comparison
        headers = ['Model', 'Test Loss']
        data = [
            ['GCN', '0.0186 ± 0.0035'],
            ['GraphSAGE', '0.0146 ± 0.0036'],
            ['GAT', '0.0232 ± 0.0051'],
        ]
        title = 'GNN Model Test Loss Comparison'
        figsize = (6, 3)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4A90E2')  # Blue header
        cell.set_text_props(weight='bold', color='white')
        cell.set_edgecolor('white')
        cell.set_linewidth(1.5)
    
    # Style data rows
    for i in range(len(data)):
        for j in range(len(headers)):
            cell = table[(i + 1, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F5F5F5')  # Light gray for alternating rows
            else:
                cell.set_facecolor('white')
            cell.set_edgecolor('#CCCCCC')
            cell.set_linewidth(1)
    
    # Style the table
    table.auto_set_column_width(list(range(len(headers))))
    
    # Add title
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved table to: {output_path}")


if __name__ == "__main__":
    output_dir = Path("reports/distribution_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create architecture table
    output_path1 = output_dir / "gnn_models_table.png"
    create_table_image(output_path1, table_type='architecture')
    
    # Create test loss table
    output_path2 = output_dir / "gnn_test_loss_table.png"
    create_table_image(output_path2, table_type='test_loss')


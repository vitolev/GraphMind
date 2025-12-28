# Professional Visualization Guide: Stanford-Style Academic Diagrams

This guide provides strategies and code examples for creating publication-quality visualizations for the GraphMind project, inspired by Stanford CS224W and other top-tier academic publications.

---

## Design Principles

### Key Characteristics of Stanford-Style Visualizations

1. **Clean, Minimalist Aesthetics**
   - Simple backgrounds (white or light gray)
   - High contrast between elements
   - Ample white space
   - No unnecessary decorations

2. **Professional Color Schemes**
   - Use ColorBrewer palettes (colorblind-friendly)
   - Limited palette (3-5 colors for main elements)
   - Consistent color meanings across figures
   - Subtle gradients when needed

3. **Typography**
   - Sans-serif fonts (Arial, Helvetica, or similar)
   - Clear hierarchy (titles, labels, annotations)
   - Appropriate font sizes (12-14pt for labels, 16-18pt for titles)

4. **High Resolution**
   - 300 DPI minimum for publication
   - Vector formats (SVG, PDF) when possible
   - PNG with high DPI for raster

5. **Clear Annotations**
   - Descriptive axis labels
   - Units where appropriate
   - Clear legends
   - Text annotations for key points

---

## Tools and Libraries

### Recommended Python Libraries

```python
# Core plotting
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import seaborn as sns

# Graph visualization
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

# Professional styling
import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

# High-quality output
import matplotlib
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
```

### Optional Advanced Tools

- **Manim** (for animations)
- **Graphviz** (for complex flowcharts)
- **Plotly** (for interactive visualizations)
- **Adobe Illustrator / Inkscape** (for final touches)

---

## Visualization Types

### 1. Pipeline Flow Diagram

**Purpose**: Show the 6-step iterative process

**Style**: Clean flowchart with numbered steps, arrows, and clear labels

```python
"""
Create a professional pipeline flow diagram
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_pipeline_diagram(output_path='pipeline_flow.png'):
    """
    Create a Stanford-style pipeline flow diagram showing the 6-step process.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Stanford-style color palette (ColorBrewer Set2)
    colors = {
        'step': '#66C2A5',      # Teal green
        'arrow': '#8DA0CB',     # Blue-purple
        'highlight': '#FC8D62', # Orange-red
        'background': '#F7F7F7', # Light gray
        'text': '#1F1F1F'       # Dark gray/black
    }
    
    # Step boxes
    step_configs = [
        {'name': '1. Graph\nGeneration', 'x': 1.5, 'y': 6, 'width': 1.5, 'height': 1},
        {'name': '2. GNN\nPrediction', 'x': 4, 'y': 6, 'width': 1.5, 'height': 1},
        {'name': '3. Candidate\nSelection', 'x': 6.5, 'y': 6, 'width': 1.5, 'height': 1},
        {'name': '4. LLM\nEvaluation', 'x': 4, 'y': 4, 'width': 1.5, 'height': 1},
        {'name': '5. Data\nUpdate', 'x': 1.5, 'y': 2, 'width': 1.5, 'height': 1},
        {'name': '6. GNN\nRetraining', 'x': 6.5, 'y': 2, 'width': 1.5, 'height': 1},
    ]
    
    # Draw steps
    for i, step in enumerate(step_configs):
        # Box with rounded corners
        box = FancyBboxPatch(
            (step['x'], step['y']),
            step['width'], step['height'],
            boxstyle="round,pad=0.1",
            linewidth=2,
            edgecolor=colors['text'],
            facecolor=colors['step'],
            zorder=2
        )
        ax.add_patch(box)
        
        # Step number circle
        circle = Circle(
            (step['x'] + 0.2, step['y'] + step['height'] - 0.2),
            0.15,
            facecolor='white',
            edgecolor=colors['text'],
            linewidth=2,
            zorder=3
        )
        ax.add_patch(circle)
        ax.text(
            step['x'] + 0.2, step['y'] + step['height'] - 0.2,
            f"{i+1}",
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            color=colors['text'],
            zorder=4
        )
        
        # Step label
        ax.text(
            step['x'] + step['width']/2, step['y'] + step['height']/2,
            step['name'],
            ha='center', va='center',
            fontsize=11, fontweight='bold',
            color='white',
            zorder=4
        )
    
    # Arrows (flow direction)
    arrows = [
        {'start': (3, 6.5), 'end': (4, 6.5), 'style': '->'},
        {'start': (5.5, 6.5), 'end': (6.5, 6.5), 'style': '->'},
        {'start': (7.25, 6), 'end': (7.25, 4.5), 'style': '->'},
        {'start': (4.75, 4), 'end': (4.75, 2.5), 'style': '->'},
        {'start': (2.25, 2), 'end': (6.5, 2), 'style': '->'},
        {'start': (1.5, 2.5), 'end': (1.5, 6), 'style': '->', 'style': '-', 'dashed': True},  # Loop back
    ]
    
    for arrow in arrows:
        arrow_patch = FancyArrowPatch(
            arrow['start'], arrow['end'],
            arrowstyle='->',
            mutation_scale=20,
            linewidth=2.5,
            color=colors['arrow'],
            zorder=1
        )
        ax.add_patch(arrow_patch)
    
    # Loop-back annotation
    ax.annotate(
        'Iteration Loop',
        xy=(1.5, 4), xytext=(0.5, 4.5),
        arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2, linestyle='--'),
        fontsize=10, style='italic',
        color=colors['text']
    )
    
    # Central dataset box
    dataset_box = FancyBboxPatch(
        (4, 2.75), 1.5, 0.5,
        boxstyle="round,pad=0.05",
        linewidth=2,
        edgecolor=colors['highlight'],
        facecolor='white',
        linestyle='--',
        zorder=2
    )
    ax.add_patch(dataset_box)
    ax.text(
        4.75, 3,
        'Training Dataset',
        ha='center', va='center',
        fontsize=10, fontweight='bold',
        color=colors['highlight'],
        zorder=4
    )
    
    # Title
    ax.text(
        5, 7.5,
        'GraphMind: 6-Step Iterative Pipeline',
        ha='center', va='center',
        fontsize=18, fontweight='bold',
        color=colors['text'],
        zorder=5
    )
    
    # Statistics annotations
    stats_text = (
        "200K graphs generated → 10 selected → 5 problems each\n"
        "99.995% evaluation reduction per iteration"
    )
    ax.text(
        5, 0.5,
        stats_text,
        ha='center', va='center',
        fontsize=9,
        color=colors['text'],
        style='italic',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=colors['arrow'], alpha=0.8),
        zorder=5
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Pipeline diagram saved to: {output_path}")

# Usage
if __name__ == "__main__":
    create_pipeline_diagram('reports/blog_images/pipeline_flow_stanford_style.png')
```

---

### 2. Graph Topology Visualization

**Purpose**: Show example multi-agent graph structures

**Style**: Clean node-link diagrams with color-coded node types

```python
"""
Create professional graph topology visualizations
"""
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_graph_topology_diagram(nodes, edges, node_types, output_path='graph_topology.png', 
                                   title="Example Multi-Agent Graph Topology"):
    """
    Create a Stanford-style graph visualization with color-coded node types.
    
    Args:
        nodes: List of (node_id, node_type) tuples
        edges: List of (from_id, to_id) tuples
        node_types: Dict mapping node_type to display name and color
        output_path: Output file path
        title: Figure title
    """
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for node_id, node_type in nodes:
        G.add_node(node_id, node_type=node_type)
    
    # Add edges
    for src, dst in edges:
        G.add_edge(src, dst)
    
    # Stanford color palette for node types
    node_colors_map = {
        'START': '#E8E8E8',      # Light gray
        'END': '#E8E8E8',        # Light gray
        'Solver': '#66C2A5',     # Teal
        'Python_solver': '#FC8D62',  # Orange
        'Validator': '#8DA0CB',  # Blue-purple
        'Extract_topic': '#E78AC3',  # Pink
        'Decompose': '#A6D854',  # Green
        'Split': '#FFD92F',      # Yellow
        'Combine_all': '#E5C494', # Tan
        'Explain': '#B3B3B3',    # Gray
    }
    
    # Layout using hierarchical layout (good for directed graphs)
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    # Alternative: hierarchical layout
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Draw edges first (behind nodes)
    nx.draw_networkx_edges(
        G, pos,
        edge_color='#666666',
        arrows=True,
        arrowsize=20,
        arrowstyle='->',
        width=2,
        alpha=0.6,
        ax=ax
    )
    
    # Draw nodes by type
    for node_type, color in node_colors_map.items():
        nodes_of_type = [n for n, d in G.nodes(data=True) if d.get('node_type') == node_type]
        if nodes_of_type:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nodes_of_type,
                node_color=color,
                node_size=2000,
                edgecolors='#1F1F1F',
                linewidths=2,
                ax=ax,
                alpha=0.9
            )
    
    # Draw labels
    labels = {n: d['node_type'].replace('_', '\n') for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=9,
        font_weight='bold',
        font_family='sans-serif',
        ax=ax
    )
    
    # Create legend
    legend_elements = []
    for node_type, color in node_colors_map.items():
        if any(n[1] == node_type for n in nodes):
            legend_elements.append(
                mpatches.Patch(facecolor=color, edgecolor='#1F1F1F', label=node_type.replace('_', ' '))
            )
    
    ax.legend(
        handles=legend_elements,
        loc='upper left',
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=10,
        title='Node Types',
        title_fontsize=12
    )
    
    # Title
    ax.text(
        0.5, 0.95,
        title,
        transform=ax.transAxes,
        ha='center', va='top',
        fontsize=16, fontweight='bold',
        color='#1F1F1F'
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Graph topology saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    # Example graph: Extract_topic → Decompose → [Solver, Solver] → Combine_all
    nodes = [
        (0, 'START'),
        (1, 'Extract_topic'),
        (2, 'Decompose'),
        (3, 'Solver'),
        (4, 'Solver'),
        (5, 'Combine_all'),
        (6, 'END')
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (2, 4),
        (3, 5), (4, 5), (5, 6)
    ]
    
    create_graph_topology_diagram(
        nodes, edges, {},
        'reports/blog_images/graph_topology_example.png',
        'Example: Topic Extraction with Parallel Solvers'
    )
```

---

### 3. GNN Architecture Diagram

**Purpose**: Illustrate how GNN processes graph structures

**Style**: Layered architecture diagram showing input → processing → output

```python
"""
Create GNN architecture visualization
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

def create_gnn_architecture_diagram(output_path='gnn_architecture.png'):
    """
    Create a Stanford-style GNN architecture diagram showing graph → GNN → prediction.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    colors = {
        'input': '#8DA0CB',      # Blue
        'process': '#66C2A5',    # Teal
        'output': '#FC8D62',     # Orange
        'arrow': '#666666',      # Gray
        'text': '#1F1F1F'        # Dark
    }
    
    # Input: Graph structure
    input_box = FancyBboxPatch(
        (0.5, 2), 2, 2,
        boxstyle="round,pad=0.15",
        linewidth=2,
        edgecolor=colors['text'],
        facecolor=colors['input'],
        alpha=0.8,
        zorder=2
    )
    ax.add_patch(input_box)
    ax.text(1.5, 3.5, 'Graph\nStructure', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white', zorder=3)
    ax.text(1.5, 2.5, 'Nodes: [Solver, ...]\nEdges: [(0,1), ...]', 
            ha='center', va='center', fontsize=9, color='white', zorder=3)
    
    # Arrow 1
    arrow1 = FancyArrowPatch((2.5, 3), (3.5, 3), arrowstyle='->',
                             mutation_scale=25, linewidth=3, color=colors['arrow'], zorder=1)
    ax.add_patch(arrow1)
    
    # GNN Layers
    layers = [
        {'name': 'Node\nEmbedding', 'x': 4, 'y': 3.5, 'width': 1.2, 'height': 1},
        {'name': 'GCN/GAT\nLayers', 'x': 5.5, 'y': 3.5, 'width': 1.2, 'height': 1},
        {'name': 'Graph\nPooling', 'x': 7, 'y': 3.5, 'width': 1.2, 'height': 1},
    ]
    
    for layer in layers:
        box = FancyBboxPatch(
            (layer['x'], layer['y']), layer['width'], layer['height'],
            boxstyle="round,pad=0.1",
            linewidth=2,
            edgecolor=colors['text'],
            facecolor=colors['process'],
            zorder=2
        )
        ax.add_patch(box)
        ax.text(layer['x'] + layer['width']/2, layer['y'] + layer['height']/2,
                layer['name'], ha='center', va='center',
                fontsize=10, fontweight='bold', color='white', zorder=3)
    
    # Arrows between layers
    for i in range(len(layers)-1):
        arrow = FancyArrowPatch(
            (layers[i]['x'] + layers[i]['width'], layers[i]['y'] + layers[i]['height']/2),
            (layers[i+1]['x'], layers[i+1]['y'] + layers[i+1]['height']/2),
            arrowstyle='->', mutation_scale=20, linewidth=2.5, color=colors['arrow'], zorder=1
        )
        ax.add_patch(arrow)
    
    # Output: Prediction
    output_box = FancyBboxPatch(
        (8.5, 2.5), 1.5, 1,
        boxstyle="round,pad=0.15",
        linewidth=2,
        edgecolor=colors['text'],
        facecolor=colors['output'],
        zorder=2
    )
    ax.add_patch(output_box)
    ax.text(9.25, 3, 'Performance\nPrediction', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white', zorder=3)
    ax.text(9.25, 2.65, 'Score: 0.85', ha='center', va='center',
            fontsize=9, color='white', zorder=3)
    
    # Arrow to output
    arrow_out = FancyArrowPatch((8.2, 3), (8.5, 3), arrowstyle='->',
                                mutation_scale=25, linewidth=3, color=colors['arrow'], zorder=1)
    ax.add_patch(arrow_out)
    
    # Title
    ax.text(5, 5.5, 'GNN Architecture: Graph → Prediction', ha='center', va='center',
            fontsize=18, fontweight='bold', color=colors['text'], zorder=5)
    
    # Annotation for graph features
    ax.annotate('Node features:\n- Node type\n- Structural\n  properties',
                xy=(1.5, 2), xytext=(0.5, 0.5),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5),
                fontsize=9, color=colors['text'],
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=colors['arrow'], alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ GNN architecture diagram saved to: {output_path}")

if __name__ == "__main__":
    create_gnn_architecture_diagram('reports/blog_images/gnn_architecture.png')
```

---

### 4. Results Comparison Charts

**Purpose**: Show performance improvements, RMSE trends, score distributions

**Style**: Clean line plots, histograms with professional styling

```python
"""
Create professional results charts in Stanford style
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def setup_stanford_style():
    """Configure matplotlib with Stanford-style settings"""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("Set2")
    
    # Professional font settings
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.labelsize': 13,
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 18,
        'figure.titleweight': 'bold',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 1.5,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
    })

def create_rmse_trend_chart(iterations, rmse_values, output_path='rmse_trend.png'):
    """
    Create a professional RMSE trend chart showing GNN prediction improvement.
    """
    setup_stanford_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot main line
    ax.plot(iterations, rmse_values, marker='o', linewidth=2.5, markersize=8,
            color='#8DA0CB', label='RMSE (GNN vs LLM)', zorder=3)
    
    # Add smooth trend line (optional)
    if len(iterations) > 3:
        z = np.polyfit(iterations, rmse_values, 2)
        p = np.poly1d(z)
        smooth_x = np.linspace(iterations.min(), iterations.max(), 100)
        ax.plot(smooth_x, p(smooth_x), '--', linewidth=2, alpha=0.6,
                color='#FC8D62', label='Trend', zorder=2)
    
    # Styling
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('RMSE', fontweight='bold')
    ax.set_title('GNN Prediction Accuracy Improvement Over Iterations', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Add improvement annotation
    if len(rmse_values) > 1:
        improvement = ((rmse_values[0] - rmse_values[-1]) / rmse_values[0]) * 100
        ax.text(0.98, 0.02, f'Improvement: {improvement:.1f}%',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#8DA0CB', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ RMSE trend chart saved to: {output_path}")

def create_score_distribution_comparison(random_scores, gnn_scores, output_path='score_distribution.png'):
    """
    Create a professional histogram comparing random vs GNN-guided sampling.
    """
    setup_stanford_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Create histogram with transparency
    bins = np.linspace(0, 1, 30)
    
    ax.hist(random_scores, bins=bins, alpha=0.6, color='#8DA0CB', 
            label=f'Random Sampling (n={len(random_scores)})', edgecolor='black', linewidth=1)
    ax.hist(gnn_scores, bins=bins, alpha=0.6, color='#FC8D62',
            label=f'GNN-Guided (n={len(gnn_scores)})', edgecolor='black', linewidth=1)
    
    # Add vertical mean lines
    ax.axvline(np.mean(random_scores), color='#8DA0CB', linestyle='--', linewidth=2.5, 
               label=f'Random Mean: {np.mean(random_scores):.3f}')
    ax.axvline(np.mean(gnn_scores), color='#FC8D62', linestyle='--', linewidth=2.5,
               label=f'GNN Mean: {np.mean(gnn_scores):.3f}')
    
    # Styling
    ax.set_xlabel('Performance Score', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Score Distribution: Random vs GNN-Guided Sampling', fontweight='bold', pad=20)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Score distribution comparison saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    # Mock data
    iterations = np.arange(1, 41)
    rmse_values = 0.3 * np.exp(-iterations/15) + 0.05 + np.random.normal(0, 0.01, 40)
    
    create_rmse_trend_chart(iterations, rmse_values, 'reports/blog_images/rmse_trend_stanford.png')
    
    # Mock score distributions
    random_scores = np.random.beta(2, 3, 100) * 0.7 + 0.3
    gnn_scores = np.random.beta(4, 2, 150) * 0.5 + 0.5
    
    create_score_distribution_comparison(random_scores, gnn_scores, 
                                        'reports/blog_images/score_distribution_stanford.png')
```

---

### 5. Algorithm Pseudocode Visualization

**Purpose**: Show algorithm flow in a clean, readable format

```python
"""
Create algorithm pseudocode diagram
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle

def create_algorithm_pseudocode(output_path='algorithm_pseudocode.png'):
    """
    Create a clean pseudocode visualization for the main pipeline algorithm.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    colors = {
        'background': '#F7F7F7',
        'header': '#8DA0CB',
        'loop': '#66C2A5',
        'step': '#E8E8E8',
        'text': '#1F1F1F'
    }
    
    # Title
    title_box = FancyBboxPatch((0.5, 11), 7, 0.8, boxstyle="round,pad=0.1",
                               linewidth=2, edgecolor=colors['text'], facecolor=colors['header'])
    ax.add_patch(title_box)
    ax.text(4, 11.4, 'GraphMind Pipeline Algorithm', ha='center', va='center',
            fontsize=16, fontweight='bold', color='white')
    
    # Pseudocode content
    code_lines = [
        "1: Initialize GNN model M",
        "2: Load training dataset D ← ∅",
        "3:",
        "4: for iteration = 1 to max_iterations:",
        "5:     // Step 1: Generate candidate graphs",
        "6:     G ← GenerateGraphs(200,000)",
        "7:",
        "8:     // Step 2: Predict performance",
        "9:     for graph g in G:",
        "10:        g.gnn_score ← M.predict(g)",
        "11:",
        "12:    // Step 3: Select top candidates",
        "13:    S ← SelectTopK(G, k=10)",
        "14:",
        "15:    // Step 4: Evaluate with LLMs",
        "16:    for graph g in S:",
        "17:        g.llm_score ← EvaluateWithLLM(g)",
        "18:",
        "19:    // Step 5: Update training data",
        "20:    D ← D ∪ S",
        "21:",
        "22:    // Step 6: Retrain model",
        "23:    M ← TrainGNN(D)",
        "24:",
        "25: return BestGraphs(D)"
    ]
    
    y_start = 10
    line_height = 0.35
    font_size = 10
    
    for i, line in enumerate(code_lines):
        y = y_start - i * line_height
        
        # Highlight loop
        if line.strip().startswith('for'):
            box = Rectangle((0.7, y - 0.15), 6.6, line_height + 0.05,
                           facecolor=colors['loop'], alpha=0.3, zorder=1)
            ax.add_patch(box)
        
        # Regular code line
        ax.text(1, y, line, ha='left', va='center',
                fontsize=font_size, fontfamily='monospace',
                color=colors['text'], zorder=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Algorithm pseudocode saved to: {output_path}")

if __name__ == "__main__":
    create_algorithm_pseudocode('reports/blog_images/algorithm_pseudocode.png')
```

---

## Complete Visualization Script

Here's a master script that creates all visualizations:

```python
"""
Master script to generate all Stanford-style visualizations for GraphMind
"""
import os
from pathlib import Path

# Create output directory
output_dir = Path('reports/blog_images')
output_dir.mkdir(parents=True, exist_ok=True)

# Import visualization functions (from above)
# ... (include all the functions defined above)

def generate_all_visualizations():
    """Generate all visualizations"""
    print("Generating Stanford-style visualizations...")
    print("=" * 60)
    
    # 1. Pipeline flow diagram
    print("\n1. Creating pipeline flow diagram...")
    create_pipeline_diagram(output_dir / 'pipeline_flow.png')
    
    # 2. Example graph topologies
    print("\n2. Creating graph topology examples...")
    # Example 1: Simple solver chain
    nodes1 = [(0, 'START'), (1, 'Solver'), (2, 'END')]
    edges1 = [(0, 1), (1, 2)]
    create_graph_topology_diagram(nodes1, edges1, {}, 
                                  output_dir / 'graph_simple.png',
                                  'Simple: Linear Solver Chain')
    
    # Example 2: Complex topology
    nodes2 = [(0, 'START'), (1, 'Extract_topic'), (2, 'Decompose'),
              (3, 'Solver'), (4, 'Solver'), (5, 'Combine_all'), (6, 'END')]
    edges2 = [(0, 1), (1, 2), (2, 3), (2, 4), (3, 5), (4, 5), (5, 6)]
    create_graph_topology_diagram(nodes2, edges2, {},
                                  output_dir / 'graph_complex.png',
                                  'Complex: Parallel Decomposition')
    
    # 3. GNN architecture
    print("\n3. Creating GNN architecture diagram...")
    create_gnn_architecture_diagram(output_dir / 'gnn_architecture.png')
    
    # 4. Algorithm pseudocode
    print("\n4. Creating algorithm pseudocode...")
    create_algorithm_pseudocode(output_dir / 'algorithm_pseudocode.png')
    
    # 5. Results charts (require data)
    print("\n5. Creating results charts...")
    # These would use actual data from your experiments
    # create_rmse_trend_chart(...)
    # create_score_distribution_comparison(...)
    
    print("\n" + "=" * 60)
    print("✓ All visualizations generated!")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    generate_all_visualizations()
```

---

## Additional Tips

### Color Palette Recommendations

**Stanford CS224W Style (ColorBrewer Set2)**:
```python
palette = {
    'primary': '#8DA0CB',    # Blue-purple
    'secondary': '#66C2A5',  # Teal-green
    'accent': '#FC8D62',     # Orange-red
    'neutral': '#E8E8E8',    # Light gray
    'text': '#1F1F1F'        # Dark gray/black
}
```

### Export Settings

Always use:
- **DPI**: 300 minimum (600 for print)
- **Format**: PDF for vector, PNG for raster
- **Bbox**: `bbox_inches='tight'` to avoid clipping
- **Facecolor**: `'white'` for clean backgrounds

### Final Touches

1. **Use LaTeX for math**: `plt.rcParams['text.usetex'] = True` (if LaTeX installed)
2. **Consistent spacing**: Use `plt.tight_layout()` or manual `subplots_adjust()`
3. **Legend placement**: Upper right or as separate panel
4. **Annotations**: Use arrows and text boxes for key insights
5. **Grid**: Subtle grid (alpha=0.3) for readability

---

## Advanced: Animation with Manim (Optional)

For animated visualizations (like Stanford's lecture videos):

```python
from manim import *

class PipelineAnimation(Scene):
    def construct(self):
        # Create animated pipeline flow
        # (Requires Manim installation: pip install manim)
        pass
```

---

This guide provides a comprehensive foundation for creating Stanford-quality visualizations. Adjust colors, layouts, and styles to match your specific needs while maintaining the core principles of clarity, professionalism, and high resolution.


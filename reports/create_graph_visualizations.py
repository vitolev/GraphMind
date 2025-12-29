"""
Create visualizations of good and bad performing graphs from the SAGE experiment.

This script:
1. Loads the training dataset from final-experiment-sage
2. Samples good (high score) and bad (low score) performing graphs
3. Visualizes them using networkx with left-to-right layout
4. Saves visualizations to reports/good-bad-graphs/
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import logging
from typing import List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set high DPI for Medium-quality images
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.facecolor'] = 'white'


def load_training_dataset(experiment_name: str = 'sage'):
    """Load training dataset from experiment folder."""
    dataset_path = Path(f"final-experiment-{experiment_name}/training_dataset.pkl")
    
    if not dataset_path.exists():
        logger.error(f"Training dataset not found: {dataset_path}")
        return None
    
    logger.info(f"Loading training dataset from: {dataset_path}")
    
    try:
        from data_management.graph_storage import GraphSet
        
        with open(dataset_path, 'rb') as f:
            training_dataset = pickle.load(f)
        
        # Extract graphs with both GNN and LLM scores
        graphs_with_scores = []
        for graph in training_dataset.graphs:
            llm_score = graph.llm_score
            gnn_score = graph.gnn_score
            if llm_score >= 0:  # Only include evaluated graphs
                graphs_with_scores.append((graph, gnn_score, llm_score))
        
        logger.info(f"Loaded {len(graphs_with_scores)} evaluated graphs")
        return graphs_with_scores
        
    except Exception as e:
        logger.error(f"Error loading training dataset: {e}")
        return None


def graph_to_networkx(graph) -> nx.DiGraph:
    """Convert our Graph object to NetworkX DiGraph."""
    G = nx.DiGraph()
    
    # Add nodes with their types
    nodes = graph.get_nodes() if hasattr(graph, 'get_nodes') else graph.nodes
    edges = graph.get_edges() if hasattr(graph, 'get_edges') else graph.edges
    
    for node_id, node_type in nodes:
        G.add_node(node_id, node_type=node_type)
    
    # Add edges
    for src, dst in edges:
        G.add_edge(src, dst)
    
    return G


def visualize_graph(graph, gnn_score: float, llm_score: float, output_path: Path, title: str = ""):
    """
    Visualize a graph with left-to-right layout.
    
    Args:
        graph: Graph object
        gnn_score: GNN predicted score
        llm_score: LLM evaluated score
        output_path: Path to save the image
        title: Title for the plot
    """
    # Convert to NetworkX
    G = graph_to_networkx(graph)
    
    if len(G.nodes()) == 0:
        logger.warning(f"Empty graph, skipping visualization")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Use hierarchical layout (left-to-right)
    try:
        # Create levels based on distance from root (BFS-based hierarchical layout)
        # Find nodes with no incoming edges (roots)
        roots = [n for n in G.nodes() if G.in_degree(n) == 0]
        if not roots:
            # If no clear roots, start from first node
            roots = [list(G.nodes())[0]]
        
        # Assign levels using BFS (level = distance from root)
        levels = {}
        visited = set()
        queue = [(root, 0) for root in roots]
        
        while queue:
            node, level = queue.pop(0)
            if node not in visited:
                visited.add(node)
                # If node already has a level, use the minimum (closest to root)
                if node not in levels or level < levels[node]:
                    levels[node] = level
                
                # Add successors to queue
                for neighbor in G.successors(node):
                    if neighbor not in visited:
                        queue.append((neighbor, level + 1))
        
        # Add any unvisited nodes to the last level
        for node in G.nodes():
            if node not in levels:
                max_level = max(levels.values()) if levels else 0
                levels[node] = max_level + 1
        
        # Group nodes by level
        nodes_by_level = {}
        for node, level in levels.items():
            nodes_by_level.setdefault(level, []).append(node)
        
        # Sort levels
        sorted_levels = sorted(nodes_by_level.keys())
        max_level = max(sorted_levels) if sorted_levels else 0
        
        # Assign positions: x based on level, y distributed vertically
        pos = {}
        for level in sorted_levels:
            nodes_in_level = nodes_by_level[level]
            n_nodes_in_level = len(nodes_in_level)
            
            # X position: level position (0 to 1)
            x_pos = level / max(1, max_level)
            
            # Y positions: evenly distributed in [0.1, 0.9]
            if n_nodes_in_level == 1:
                y_positions = [0.5]
            else:
                y_positions = np.linspace(0.1, 0.9, n_nodes_in_level)
            
            for i, node in enumerate(nodes_in_level):
                pos[node] = (x_pos, y_positions[i])
        
    except Exception as e:
        logger.warning(f"Could not create hierarchical layout: {e}, using spring layout")
        pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
    
    # Node colors by type
    node_colors = {
        'Solver': '#3498DB',           # Blue
        'Python_solver': '#E74C3C',    # Red
        'Validator': '#2ECC71',        # Green
        'Extract_topic': '#9B59B6',    # Purple
        'Decompose': '#F39C12',        # Orange
        'Split': '#1ABC9C',            # Teal
        'Combine_all': '#E67E22',      # Dark Orange
        'Explain': '#95A5A6',          # Gray
        'START': '#34495E',            # Dark Gray
        'END': '#34495E'               # Dark Gray
    }
    
    # Draw nodes
    node_types = nx.get_node_attributes(G, 'node_type')
    colors = [node_colors.get(node_types.get(node, 'Solver'), '#95A5A6') for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=2000, 
                          alpha=0.9, ax=ax, linewidths=2, edgecolors='white')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                          arrowsize=20, arrowstyle='->', width=2, alpha=0.6, ax=ax)
    
    # Draw labels
    labels = {}
    for node in G.nodes():
        node_type = node_types.get(node, 'Unknown')
        # Shorten long names
        if node_type == 'Python_solver':
            labels[node] = f'Python\n{node}'
        elif node_type == 'Extract_topic':
            labels[node] = f'Topic\n{node}'
        elif node_type == 'Combine_all':
            labels[node] = f'Combine\n{node}'
        else:
            labels[node] = f'{node_type}\n{node}'
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax)
    
    # Title with both scores
    if not title:
        title_text = f"GNN: {gnn_score:.4f} | LLM: {llm_score:.4f}"
    else:
        title_text = f"{title}\nGNN: {gnn_score:.4f} | LLM: {llm_score:.4f}"
    ax.set_title(title_text, fontsize=12, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Saved graph visualization to: {output_path}")


def categorize_graphs(graphs_with_scores, gnn_threshold=0.7, llm_threshold=0.7):
    """
    Categorize graphs into 4 groups based on GNN and LLM scores.
    
    Args:
        graphs_with_scores: List of (graph, gnn_score, llm_score) tuples
        gnn_threshold: Threshold for high/low GNN prediction
        llm_threshold: Threshold for high/low LLM performance
    
    Returns:
        Dictionary with 4 categories
    """
    categories = {
        'high_gnn_high_llm': [],  # High prediction, high performance
        'high_gnn_low_llm': [],   # High prediction, low performance (overestimation)
        'low_gnn_high_llm': [],   # Low prediction, high performance (underestimation)
        'low_gnn_low_llm': []     # Low prediction, low performance
    }
    
    for graph, gnn_score, llm_score in graphs_with_scores:
        if gnn_score >= gnn_threshold and llm_score >= llm_threshold:
            categories['high_gnn_high_llm'].append((graph, gnn_score, llm_score))
        elif gnn_score >= gnn_threshold and llm_score < llm_threshold:
            categories['high_gnn_low_llm'].append((graph, gnn_score, llm_score))
        elif gnn_score < gnn_threshold and llm_score >= llm_threshold:
            categories['low_gnn_high_llm'].append((graph, gnn_score, llm_score))
        else:
            categories['low_gnn_low_llm'].append((graph, gnn_score, llm_score))
    
    return categories


def main():
    """Main function to create graph visualizations."""
    logger.info("=" * 60)
    logger.info("CREATING GRAPH VISUALIZATIONS")
    logger.info("=" * 60)
    
    # Load training dataset
    graphs_with_scores = load_training_dataset('sage')
    
    if graphs_with_scores is None or len(graphs_with_scores) == 0:
        logger.error("No graphs loaded!")
        return
    
    # Categorize graphs
    logger.info("\nCategorizing graphs by GNN prediction and LLM performance...")
    categories = categorize_graphs(graphs_with_scores, gnn_threshold=0.7, llm_threshold=0.7)
    
    logger.info(f"\nCategory counts:")
    logger.info(f"  High GNN + High LLM: {len(categories['high_gnn_high_llm'])}")
    logger.info(f"  High GNN + Low LLM: {len(categories['high_gnn_low_llm'])}")
    logger.info(f"  Low GNN + High LLM: {len(categories['low_gnn_high_llm'])}")
    logger.info(f"  Low GNN + Low LLM: {len(categories['low_gnn_low_llm'])}")
    
    # Sample graphs from each category
    num_samples_per_category = 5  # Number of graphs to visualize from each category
    
    # Create output directory
    output_dir = Path("reports/good-bad-graphs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Category names and titles
    category_info = {
        'high_gnn_high_llm': {
            'title': 'High GNN Prediction, High LLM Performance',
            'prefix': 'high_gnn_high_llm'
        },
        'high_gnn_low_llm': {
            'title': 'High GNN Prediction, Low LLM Performance',
            'prefix': 'high_gnn_low_llm'
        },
        'low_gnn_high_llm': {
            'title': 'Low GNN Prediction, High LLM Performance',
            'prefix': 'low_gnn_high_llm'
        },
        'low_gnn_low_llm': {
            'title': 'Low GNN Prediction, Low LLM Performance',
            'prefix': 'low_gnn_low_llm'
        }
    }
    
    # Visualize graphs from each category
    for category_name, info in category_info.items():
        category_graphs = categories[category_name]
        
        if len(category_graphs) == 0:
            logger.warning(f"No graphs in category: {category_name}")
            continue
        
        # Sort by LLM score (descending) for consistency
        category_graphs.sort(key=lambda x: x[2], reverse=True)
        
        # Sample graphs
        num_to_sample = min(num_samples_per_category, len(category_graphs))
        sampled_graphs = category_graphs[:num_to_sample]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Visualizing {category_name} ({num_to_sample} graphs)")
        logger.info(f"{'='*60}")
        
        for i, (graph, gnn_score, llm_score) in enumerate(sampled_graphs):
            output_path = output_dir / f"{info['prefix']}_graph_{i+1}_gnn_{gnn_score:.4f}_llm_{llm_score:.4f}.png"
            visualize_graph(graph, gnn_score, llm_score, output_path, 
                           title=info['title'])
    
    logger.info("\n" + "=" * 60)
    logger.info("ALL VISUALIZATIONS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()


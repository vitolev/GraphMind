"""
Diagnostic tools for analyzing training data and experiment results.

This module provides functions to:
- Load training dataset
- Visualize predictions vs actual scores
- Show best performing graphs
- Generate diagnostic reports
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from typing import List, Optional, Tuple
from config.settings import Config
from data_management.graph_storage import load_training_dataset, Graph, GraphSet
# Note: build_langgraph not needed for diagnostics - removed import


def visualize_predictions_vs_actual(
    training_dataset: GraphSet,
    output_dir: Path,
    logger: logging.Logger,
    top_n: int = 20
) -> Path:
    """
    Create visualization comparing GNN predictions vs LLM actual scores.
    
    Args:
        training_dataset: GraphSet containing training data
        output_dir: Directory to save visualizations
        logger: Logger instance
        top_n: Number of top graphs to highlight
    
    Returns:
        Path to saved figure
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    graphs = training_dataset.get_all()
    if len(graphs) == 0:
        logger.warning("Training dataset is empty, cannot create visualization")
        return None
    
    gnn_scores = [g.get_gnn_score() for g in graphs]
    llm_scores = [g.get_llm_score() for g in graphs]
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Data Diagnostics: GNN Predictions vs LLM Actual Scores', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot: Predictions vs Actual
    ax1 = axes[0, 0]
    ax1.scatter(gnn_scores, llm_scores, alpha=0.6, s=50)
    ax1.plot([0, 1], [0, 1], 'r--', label='Perfect prediction', linewidth=2)
    ax1.set_xlabel('GNN Predicted Score', fontsize=12)
    ax1.set_ylabel('LLM Actual Score', fontsize=12)
    ax1.set_title('Predictions vs Actual Scores', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Calculate and display metrics
    errors = np.array(llm_scores) - np.array(gnn_scores)
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    correlation = np.corrcoef(gnn_scores, llm_scores)[0, 1] if len(gnn_scores) > 1 else 0.0
    
    textstr = f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nCorrelation: {correlation:.4f}\nN={len(graphs)}'
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Highlight top N graphs
    if len(graphs) >= top_n:
        # Sort by LLM score (actual performance)
        sorted_graphs = sorted(enumerate(graphs), key=lambda x: x[1].get_llm_score(), reverse=True)
        top_indices = [i for i, _ in sorted_graphs[:top_n]]
        top_gnn = [gnn_scores[i] for i in top_indices]
        top_llm = [llm_scores[i] for i in top_indices]
        ax1.scatter(top_gnn, top_llm, color='red', s=100, marker='*', 
                   label=f'Top {top_n} graphs', zorder=5)
        ax1.legend()
    
    # 2. Error distribution
    ax2 = axes[0, 1]
    ax2.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero error')
    ax2.axvline(x=np.mean(errors), color='g', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
    ax2.set_xlabel('Error (Actual - Predicted)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Prediction Error Distribution', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Score distribution
    ax3 = axes[1, 0]
    ax3.hist(llm_scores, bins=50, alpha=0.5, label='LLM Actual', color='blue')
    ax3.hist(gnn_scores, bins=50, alpha=0.5, label='GNN Predicted', color='orange')
    ax3.set_xlabel('Score', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Score Distribution Comparison', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Score statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_data = {
        'Metric': ['Mean', 'Std', 'Min', 'Max', 'Median'],
        'GNN Predicted': [
            f"{np.mean(gnn_scores):.4f}",
            f"{np.std(gnn_scores):.4f}",
            f"{np.min(gnn_scores):.4f}",
            f"{np.max(gnn_scores):.4f}",
            f"{np.median(gnn_scores):.4f}"
        ],
        'LLM Actual': [
            f"{np.mean(llm_scores):.4f}",
            f"{np.std(llm_scores):.4f}",
            f"{np.min(llm_scores):.4f}",
            f"{np.max(llm_scores):.4f}",
            f"{np.median(llm_scores):.4f}"
        ],
        'Error': [
            f"{np.mean(errors):.4f}",
            f"{np.std(errors):.4f}",
            f"{np.min(errors):.4f}",
            f"{np.max(errors):.4f}",
            f"{np.median(errors):.4f}"
        ]
    }
    
    stats_df = pd.DataFrame(stats_data)
    table = ax4.table(cellText=stats_df.values, colLabels=stats_df.columns,
                     cellLoc='center', loc='center', colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax4.set_title('Statistics Summary', fontsize=14, pad=20)
    
    plt.tight_layout()
    
    fig_path = output_dir / "predictions_vs_actual.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved predictions vs actual visualization to {fig_path}")
    return fig_path


def visualize_graph_structure(
    graph: Graph,
    output_path: Path,
    title: Optional[str] = None,
    show_scores: bool = True
) -> None:
    """
    Visualize a single graph structure.
    
    Args:
        graph: Graph object to visualize
        output_path: Path to save the figure
        title: Optional title for the plot
        show_scores: Whether to show GNN and LLM scores in title
    """
    G = nx.DiGraph()
    
    # Add nodes with labels
    for node_id, node_type in graph.get_nodes():
        G.add_node(node_id, label=node_type)
    
    # Add edges
    G.add_edges_from(graph.get_edges())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use hierarchical layout for better visualization
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except:
        try:
            pos = nx.spring_layout(G, k=2, iterations=50)
        except:
            pos = nx.planar_layout(G)
    
    # Draw nodes with colors based on node type
    node_colors = {}
    node_type_colors = {
        'START': 'green',
        'END': 'red',
        'Solver': 'blue',
        'Python_solver': 'cyan',
        'Combine_all': 'orange',
        'Combine_any': 'orange',
        'Split': 'purple',
        'Decompose': 'brown',
        'Validator': 'pink',
        'Extract_topic': 'yellow',
        'Explain': 'gray'
    }
    
    for node_id, node_type in graph.get_nodes():
        node_colors[node_id] = node_type_colors.get(node_type, 'lightblue')
    
    node_colors_list = [node_colors.get(node, 'lightblue') for node in G.nodes()]
    
    # Draw graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors_list, node_size=1000, ax=ax)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20, edge_color='gray', ax=ax)
    
    # Create labels with node type
    labels = {node_id: f"{node_id}\n{node_type}" for node_id, node_type in graph.get_nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    # Set title
    if title is None:
        title = "Graph Structure"
    if show_scores:
        title += f"\nGNN: {graph.get_gnn_score():.4f} | LLM: {graph.get_llm_score():.4f}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_best_graphs(
    training_dataset: GraphSet,
    output_dir: Path,
    logger: logging.Logger,
    top_n: int = 10
) -> List[Path]:
    """
    Visualize the top N best performing graphs.
    
    Args:
        training_dataset: GraphSet containing training data
        output_dir: Directory to save visualizations
        logger: Logger instance
        top_n: Number of top graphs to visualize
    
    Returns:
        List of paths to saved figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = output_dir / "best_graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    
    graphs = training_dataset.get_all()
    if len(graphs) == 0:
        logger.warning("Training dataset is empty, cannot visualize graphs")
        return []
    
    # Sort by LLM score (actual performance)
    sorted_graphs = sorted(graphs, key=lambda g: g.get_llm_score(), reverse=True)
    top_graphs = sorted_graphs[:top_n]
    
    saved_paths = []
    
    for idx, graph in enumerate(top_graphs):
        rank = idx + 1
        title = f"Top {rank} Graph (LLM Score: {graph.get_llm_score():.4f})"
        filename = f"top_{rank}_graph_llm_{graph.get_llm_score():.4f}.png"
        filepath = graphs_dir / filename
        
        visualize_graph_structure(graph, filepath, title, show_scores=True)
        saved_paths.append(filepath)
        
        logger.info(f"Saved top {rank} graph visualization: {filepath}")
    
    return saved_paths


def create_diagnostic_report(
    config: Config,
    logger: Optional[logging.Logger] = None,
    top_n_graphs: int = 10
) -> Path:
    """
    Create a comprehensive diagnostic report for the experiment.
    
    This function:
    1. Loads the training dataset
    2. Creates predictions vs actual visualization
    3. Visualizes top performing graphs
    4. Saves all outputs to the experiment's analytics directory
    
    Args:
        config: Configuration object
        logger: Optional logger instance
        top_n_graphs: Number of top graphs to visualize
    
    Returns:
        Path to the diagnostics output directory
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    
    # Setup output directory
    output_dir = config.analytics_dir / config.experiment_name / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating diagnostic report in: {output_dir}")
    
    # Load training dataset
    logger.info("Loading training dataset...")
    training_dataset = load_training_dataset(config.data_dir, logger)
    
    if training_dataset.size() == 0:
        logger.warning("Training dataset is empty. Cannot create diagnostics.")
        return output_dir
    
    logger.info(f"Loaded {training_dataset.size()} graphs from training dataset")
    
    # 1. Create predictions vs actual visualization
    logger.info("Creating predictions vs actual visualization...")
    visualize_predictions_vs_actual(training_dataset, output_dir, logger, top_n=top_n_graphs)
    
    # 2. Visualize best graphs
    logger.info(f"Visualizing top {top_n_graphs} graphs...")
    visualize_best_graphs(training_dataset, output_dir, logger, top_n=top_n_graphs)
    
    # 3. Create summary statistics CSV
    logger.info("Creating summary statistics...")
    graphs = training_dataset.get_all()
    
    summary_data = []
    for idx, graph in enumerate(graphs):
        summary_data.append({
            'graph_index': idx,
            'gnn_predicted_score': graph.get_gnn_score(),
            'llm_actual_score': graph.get_llm_score(),
            'error': graph.get_llm_score() - graph.get_gnn_score(),
            'absolute_error': abs(graph.get_llm_score() - graph.get_gnn_score()),
            'num_nodes': len(graph.get_nodes()),
            'num_edges': len(graph.get_edges()),
            'evaluation_time': graph.get_time_evaluating()
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('llm_actual_score', ascending=False)
    
    csv_path = output_dir / "graph_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    logger.info(f"Saved graph summary to {csv_path}")
    
    logger.info(f"✅ Diagnostic report complete! Outputs saved to: {output_dir}")
    logger.info(f"   - Predictions vs actual: {output_dir / 'predictions_vs_actual.png'}")
    logger.info(f"   - Top {top_n_graphs} graphs: {output_dir / 'best_graphs/'}")
    logger.info(f"   - Summary CSV: {csv_path}")
    
    return output_dir


if __name__ == "__main__":
    """
    Run diagnostics from command line.
    
    Usage:
        python -m post_processing.diagnostics
    """
    import logging
    from config.settings import Config
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load config
    config_path = Path("config/experiment_config.yaml")
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        exit(1)
    
    config = Config.from_yaml(config_path)
    
    # Create diagnostics
    output_dir = create_diagnostic_report(config, logger, top_n_graphs=10)
    print(f"\n✅ Diagnostics complete! Check: {output_dir}")


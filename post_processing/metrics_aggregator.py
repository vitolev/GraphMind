import logging
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path


def flatten_metrics_dict(metrics_dict: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    flattened = {}
    
    for key, value in metrics_dict.items():
        new_key = f"{prefix}{key}" if prefix else key
        
        if isinstance(value, dict):
            # Recursively flatten nested dicts
            nested_flat = flatten_metrics_dict(value, new_key + '_')
            flattened.update(nested_flat)
        elif isinstance(value, (list, tuple)):
            # Skip lists/tuples, just store length if needed
            flattened[new_key] = len(value)
        else:
            # Store scalar values
            flattened[new_key] = value
    
    return flattened


def create_metrics_dataframe(
    metrics_history: List[Dict[str, Any]],
    config: Any,
    logger: logging.Logger,
    selected_metrics: List[str] = None
) -> pd.DataFrame:
    
    # Flatten each iteration's metrics
    flattened_records = []
    for iteration_metrics in metrics_history:
        flat = flatten_metrics_dict(iteration_metrics)
        flattened_records.append(flat)
    
    # Create DataFrame
    df = pd.DataFrame(flattened_records)
    
    # Filter to selected metrics if provided
    if selected_metrics:
        # Add 'iteration_num' and 'timestamp' if they exist
        base_cols = ['iteration_num', 'timestamp']
        cols_to_keep = [col for col in base_cols if col in df.columns]
        cols_to_keep.extend([col for col in selected_metrics if col in df.columns])
        df = df[[col for col in cols_to_keep if col in df.columns]]
    
    logger.info(f"Created metrics dataframe with {len(df)} iterations and {len(df.columns)} metrics")
    logger.debug(f"Metrics columns: {list(df.columns)}")
    
    return df


def save_metrics_dataframe(
    df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
    filename: str = 'pipeline_metrics.csv'
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    
    df.to_csv(output_path, index=False)
    logger.info(f"Saved metrics dataframe to {output_path}")
    
    return output_path


def compute_iteration_summary(
    df: pd.DataFrame,
    logger: logging.Logger
) -> Dict[str, Any]:
    summary = {
        'total_iterations': len(df),
        'total_graphs_generated': int(df['loop_total_graphs_generated'].sum()) if 'loop_total_graphs_generated' in df.columns else 0,
        'total_graphs_evaluated': int(df['loop_total_graphs_evaluated'].sum()) if 'loop_total_graphs_evaluated' in df.columns else 0,
    }
    
    # Add mean scores
    if 'loop_best_predicted_score' in df.columns:
        summary['mean_best_predicted'] = float(df['loop_best_predicted_score'].mean())
    if 'loop_best_actual_score' in df.columns:
        summary['mean_best_actual'] = float(df['loop_best_actual_score'].mean())
    if 'step_4_evaluation_rmse_gnn_vs_llm' in df.columns:
        summary['mean_rmse_gnn_vs_llm'] = float(df['step_4_evaluation_rmse_gnn_vs_llm'].mean())
    
    logger.info(f"Iteration summary: {summary}")
    return summary
import logging
import pandas as pd
from pathlib import Path
from config.settings import Config
from post_processing.metrics_aggregator import save_metrics_dataframe, compute_iteration_summary


def run_analytics(metrics_df: pd.DataFrame, config: Config, logger: logging.Logger) -> None:
    """
    Run post-processing analysis based on config settings.
    
    Creates subdirectories and files based on what's enabled in config.
    All outputs are saved to logs/analytics/{experiment_name}/ folder.
    """
    
    # Use experiment-specific directory for all analytics outputs
    analytics_dir = config.analytics_dir / config.experiment_name
    analytics_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Analytics output directory: {analytics_dir}")
    
    # Save metrics CSV
    save_metrics_dataframe(metrics_df, analytics_dir, logger)
    
    # Compute and log summary
    summary = compute_iteration_summary(metrics_df, logger)

    plot_pipeline_metrics(metrics_df, analytics_dir, logger)
    # TODO: Add visualization functions here based on config
    # if config.generate_plots:
    #     generate_plots(metrics_df, analytics_dir, logger)
    # if config.generate_report:
    #     generate_report(metrics_df, summary, analytics_dir, logger)


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_pipeline_metrics(metrics_df: pd.DataFrame, analytics_dir: Path, logger):
    metrics = [
        ("step_2_prediction_best_predicted", "Best GNN Prediction"),
        ("step_2_prediction_mean_predicted", "Mean GNN Prediction"),
        ("step_4_evaluation_best_evaluated", "Best LLM Evaluation"),
        ("step_4_evaluation_worst_evaluated", "Worst LLM Evaluation"),
        ("step_4_evaluation_mean_evaluated", "Mean LLM Evaluation"),
        ("step_4_evaluation_rmse_gnn_vs_llm", "RMSE (GNN vs LLM)")
    ]

    num_plots = len(metrics)
    plt.figure(figsize=(10, 2.5 * num_plots))
    x = metrics_df["iteration_num"] if "iteration_num" in metrics_df.columns else np.arange(len(metrics_df))

    for idx, (col, title) in enumerate(metrics, 1):
        if col in metrics_df.columns:
            y = metrics_df[col]
            plt.subplot(num_plots, 1, idx)
            plt.plot(x, y, marker='o')
            plt.title(title)
            plt.xlabel("Iteration")
            plt.ylabel(title)
            if y.min() != y.max():
                plt.ylim([y.min() - 0.05 * abs(y.max() - y.min()), y.max() + 0.05 * abs(y.max() - y.min())])
            plt.grid(True)
        else:
            logger.warning(f"Metric column '{col}' not found in metrics_df.")

    plt.tight_layout()
    fig_path = analytics_dir / "metric_trends.png"
    plt.savefig(fig_path)
    plt.close()
    logger.info(f"Saved metric visualization to {fig_path}")


import sys
import os

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from data_management.graph_storage import GraphSet
from gnn_models.model_manager import initialize_gnn_model, predict_batch_performance
from pathlib import Path
import pickle
from typing import List, Dict, Any
import numpy as np
from config.settings import Config
import logging

def load_testing_dataset() -> tuple[GraphSet, GraphSet]:
    """
    Load old and new training datasets from pickle files. The test set is formed by the graphs
    that are present in the new dataset but not in the old dataset.

    Returns:
        tuple[GraphSet, GraphSet]: A tuple containing the new (test) dataset and the old (train + val) dataset.
    """
    dataset = GraphSet()
    dataset_path = Path("./data") / "training_dataset_new.pkl"
    
    if dataset_path.exists():
        try:
            with open(dataset_path, 'rb') as f:
                loaded = pickle.load(f)
            dataset = loaded
        except Exception as e:
            print(f"Error loading training dataset: {e}")
    else:
        print("Training dataset file does not exist. Returning empty dataset.")
    
    dataset_old = GraphSet()
    dataset_old_path = Path("./data") / "training_dataset_old.pkl"

    if dataset_old_path.exists():
        try:
            with open(dataset_old_path, 'rb') as f:
                loaded_old = pickle.load(f)
            dataset_old = loaded_old
        except Exception as e:
            print(f"Error loading old training dataset: {e}")
    else:
        print("Old training dataset file does not exist. Returning empty dataset.")

    dataset_new = GraphSet()

    new_graphs = dataset.get_all()
    for graph in new_graphs:
        if not dataset_old.contains(graph):
            dataset_new.add_graph(graph)

    return dataset_new, dataset_old

models = ["gcn", "graph_sage", "gat"]
hyperparameters = {
    "gcn": {"epochs": 300, "hidden_dim": 16, "num_layers": 3, "dropout": 0},
    "graph_sage": {"epochs": 300, "hidden_dim": 32, "num_layers": 3, "dropout": 0.2},
    "gat": {"epochs": 300, "hidden_dim": 8, "num_layers": 4, "dropout": 0.1}
}


config_path = Path("config/experiment_config.yaml")
config = Config.from_yaml(config_path)

results = {
    "gcn": [],
    "graph_sage": [],
    "gat": []
}

for i in range(10):
    print(f"--- Iteration {i+1}/10 ---")
    for model_name in models:
        print(f"Evaluating model: {model_name}")
        dataset_new, dataset_old = load_testing_dataset()

        config.gnn_model_type = model_name
        for param, value in hyperparameters[model_name].items():
            setattr(config, param, value)

        logger = logging.getLogger("gnn_multiagent")
        model = initialize_gnn_model(config, logger)

        model.fit(dataset_old.to_pyg(config))

        _, _ = predict_batch_performance(
            config,
            logger,
            model,
            dataset_new
        )

        predictions_new = []
        true_values_new = []

        for graph in dataset_new.get_all():
            prediction = graph.get_gnn_score()
            true = graph.get_llm_score()

            predictions_new.append(prediction)
            true_values_new.append(true)
        
        predictions_new = np.array(predictions_new)
        true_values_new = np.array(true_values_new)

        # Compute MSE
        mse_new = np.mean((predictions_new - true_values_new) ** 2)
        results[model_name].append(mse_new)

# Print average results
for model_name in models:
    avg_mse = np.mean(results[model_name])
    std_mse = np.std(results[model_name])
    print(f"Model: {model_name} - Average MSE over 10 runs: {avg_mse:.4f} ± {std_mse:.4f}")

# --------------------------------------------------------------------------
# Obtained results:
# Model: gcn - Average MSE over 10 runs: 0.0186 ± 0.0035
# Model: graph_sage - Average MSE over 10 runs: 0.0146 ± 0.0036
# Model: gat - Average MSE over 10 runs: 0.0232 ± 0.0051
# --------------------------------------------------------------------------







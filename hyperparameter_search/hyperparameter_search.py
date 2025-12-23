import logging
from pathlib import Path
from config.settings import Config

def setup_logging(config: Config) -> logging.Logger:
    """Set up logging to file and console"""
    config.log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("gnn_multiagent")
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(config.log_dir / f"hyperparameter_search.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def main():
    """
    Main entry point
    
    Flow:
    1. Load configuration from YAML
    2. Set up logging
    3. Create directories
    4. Run pipeline with config
    """
    # Load config
    config_path = Path("config/experiment_config.yaml")
    
    if not config_path.exists():
        print(f"Creating default config at {config_path}")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = Config()
        config.save_yaml(config_path)
    
    config = Config.from_yaml(config_path)
    
    # Set up logging
    logger = setup_logging(config)
    logger.info("=" * 60)
    logger.info("GNN-BASED HYPERPARAMETER OPTIMIZATION FRAMEWORK")
    logger.info("=" * 60)
    logger.info(f"Experiment: hyperparameter_search")
    logger.info(f"Config loaded from: {config_path}")
    logger.info(f"Data dir: {config.data_dir}")
    
    # Run pipeline
    logger.info("=" * 60)
    logger.info("STARTING PIPELINE")
    logger.info("=" * 60 + "\n")
    
    try:
        metrics_df = run_hyperparameter_search(config, logger)
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise

from data_management.graph_storage import load_training_dataset
from sklearn.model_selection import KFold
from gnn_models.model_manager import initialize_gnn_model
import numpy as np
import pandas as pd

def run_hyperparameter_search(config: Config, logger: logging.Logger):
    """
    Run hyperparameter search for GNN models.
    
    Args:
        config (Config): Configuration object.
        logger (logging.Logger): Logger for logging information.
    """
    models = ["gcn", "graph_sage", "gat"]
    hyperparams_grid = {
        "gnn_learning_rate": [0.001],
        "gnn_epochs": [50, 100, 200, 300],
        "gnn_hidden_dim": [4, 8, 16, 32],
        "gnn_num_layers": [2, 3, 4],
        "gnn_dropout": [0.0, 0.1, 0.2, 0.5]
    }
    training_dataset = load_training_dataset(config.data_dir, logger) 
    training_dataset = training_dataset.to_pyg(config)

    # k-fold cross-validation setup
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    results = []
    for i, (train_index, val_index) in enumerate(kf.split(training_dataset)):
        logger.info(f"Starting fold {i + 1}/10")
        train_subset = [training_dataset[i] for i in train_index]
        val_subset = [training_dataset[i] for i in val_index]

        for model_name in models:
            config.gnn_model_type = model_name
            for lr in hyperparams_grid["gnn_learning_rate"]:
                for epochs in hyperparams_grid["gnn_epochs"]:
                    for hidden_dim in hyperparams_grid["gnn_hidden_dim"]:
                        for num_layers in hyperparams_grid["gnn_num_layers"]:
                            for dropout in hyperparams_grid["gnn_dropout"]:
                                logger.info(
                                    f"Training {model_name} with lr={lr}, epochs={epochs}, "
                                    f"hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}"
                                )
                                # Update config with current hyperparameters
                                config.gnn_learning_rate = lr
                                config.gnn_epochs = epochs
                                config.gnn_hidden_dim = hidden_dim
                                config.gnn_num_layers = num_layers
                                config.gnn_dropout = dropout

                                # Initialize model
                                model = initialize_gnn_model(config, logger)

                                # Train model
                                train_loss = model.fit(train_subset)

                                # Validate model
                                val_preds = model.predict(val_subset)
                                val_targets = [data.y.item() for data in val_subset]
                                val_loss = np.mean((np.array(val_preds) - np.array(val_targets)) ** 2)

                                # Log results
                                result = {
                                    "model": model_name,
                                    "learning_rate": lr,
                                    "epochs": epochs,
                                    "hidden_dim": hidden_dim,
                                    "num_layers": num_layers,
                                    "dropout": dropout,
                                    "train_loss": train_loss,
                                    "val_loss": val_loss
                                }
                                results.append(result)
    
    # Save results to DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv("hyperparameter_search_results.csv", index=False)

if __name__ == "__main__":
    main()
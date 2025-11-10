import logging
import importlib
from typing import Any, List, Dict, Tuple
from config.settings import Config
import time
from data_management.graph_storage import GraphSet
import torch
import numpy as np

def initialize_gnn_model(
    config: Config,
    logger: logging.Logger
) -> Any:

    model_type = config.gnn_model_type
    logger.info(f"Initializing GNN model: {model_type}")
    
    try:
        module_name = f"gnn_models.{model_type}"
        logger.debug(f"Importing module: {module_name}")
        
        module = importlib.import_module(module_name)
        
        if not hasattr(module, 'get_model'):
            raise AttributeError(
                f"Model module '{module_name}' does not have 'get_model' function. "
                f"Each model file must define: def get_model(config, logger) -> Model"
            )
        
        model = module.get_model(config, logger)
        
        logger.info(f"  ✓ {model_type} model initialized")
        logger.debug(f"    - Model class: {type(model).__name__}")
        logger.debug(f"    - Device: {config.gnn_device}")
        
        return model
    
    except AttributeError as e:
        logger.error(str(e))
        raise
    
    except Exception as e:
        logger.error(f"Error initializing GNN model: {e}", exc_info=True)
        raise

def predict_batch_performance(
    config: Config,
    logger: logging.Logger,
    model: Any,
    generated_graphs: GraphSet
) -> Tuple[Dict[str, Any], GraphSet]:
    
    pyg_graphs = generated_graphs.to_pyg(config, type="HeteroData")
    num_graphs = len(pyg_graphs)
    
    logger.debug(f"Converted {num_graphs} graphs to HeteroData for GNN prediction")
    
    start_time = time.time()
    with torch.no_grad():
        predictions = model.predict(pyg_graphs)
    inference_time = time.time() - start_time
    
    predictions_list = list(predictions)
    
    for graph, pred in zip(generated_graphs.graphs, predictions_list):
        graph.set_gnn_score(float(pred))
    
    scores = np.array([float(p) for p in predictions_list])
    
    metrics = {
        'step_name': 'prediction',
        'duration_seconds': round(inference_time, 4),
        'num_samples': num_graphs,
        'best_predicted': float(scores.max()) if len(scores) > 0 else None,
        'worst_predicted': float(scores.min()) if len(scores) > 0 else None,
        'mean_predicted': float(scores.mean()) if len(scores) > 0 else None,
        'std_predicted': float(scores.std()) if len(scores) > 0 else None,
        'metadata': {
            'inference_time_per_graph': round(inference_time / num_graphs, 4) if num_graphs > 0 else 0,
        }
    }
    
    logger.debug(
        f"GNN prediction complete - "
        f"Best: {metrics['best_predicted']:.4f}, "
        f"Mean: {metrics['mean_predicted']:.4f}, "
        f"Time: {inference_time:.4f}s"
    )
    
    return metrics, generated_graphs

def retrain_gnn_model(
    config: Config,
    logger: logging.Logger,
    model: Any,
    training_dataset: GraphSet
) -> Any:
    
    logger.info("Retraining GNN model")
    logger.info(f"  - Training samples: {training_dataset.size()}")
    logger.info(f"  - Learning rate: {config.gnn_learning_rate}")
    
    if training_dataset.size() == 0:
        logger.warning("No training data, skipping retrain")
        return model
    
    train_data = training_dataset.to_pyg(config, type="HeteroData")
    loss = model.fit(train_data)
    
    logger.info(f"  ✓ Training complete (loss: {loss:.4f})\n")
    
    metrics = { 
        'step_name': 'retraining',
        'training_samples': training_dataset.size(),
        'final_loss': loss,
    }

    return metrics, model

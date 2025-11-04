import logging
import importlib
from typing import Any, List, Dict
from config.settings import Config
from data_management.dataset_manager import TrainingDataset

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
    gnn_graphs: List[Dict[str, Any]],
    model: Any
) -> List[Dict[str, Any]]:

    logger.debug(f"Running inference on {len(gnn_graphs)} graphs")
    
    if not gnn_graphs:
        logger.warning("No graphs provided for prediction")
        return []
    
    scores = model.predict(gnn_graphs)
    
    predictions = []
    for gnn_graph, score in zip(gnn_graphs, scores):
        predictions.append({
            'graph': gnn_graph,
            'score': float(score),
        })
    
    predictions.sort(key=lambda x: x['score'], reverse=True)
    
    logger.debug(f"  ✓ Generated {len(predictions)} predictions")
    logger.debug(
        f"    - Score range: [{predictions[-1]['score']:.4f}, {predictions['score']:.4f}]"
    )
    
    return predictions


def retrain_gnn_model(
    config: Config,
    logger: logging.Logger,
    model: Any,
    training_dataset: TrainingDataset
) -> Any:
    
    logger.info("Retraining GNN model")
    logger.info(f"  - Training samples: {training_dataset.size()}")
    logger.info(f"  - Learning rate: {config.gnn_learning_rate}")
    
    if training_dataset.size() == 0:
        logger.warning("No training data, skipping retrain")
        return model
    
    train_data = training_dataset.get_samples()
    
    loss = model.fit(train_data)
    
    logger.info(f"  ✓ Training complete (loss: {loss:.4f})\n")
    
    return model

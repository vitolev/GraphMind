import logging
import time
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from config.settings import Config
import numpy as np
from typing import Any, Dict, List

@dataclass
class IterationMetrics:
    """Metrics for a single iteration"""
    iteration_num: int
    timestamp: datetime
    graphs_generated: int
    graphs_selected: int
    graphs_evaluated: int
    training_samples_added: int
    best_predicted_score: float
    best_actual_score: float = None
    gnn_retrained: bool = False
    retrain_loss: float = None

@dataclass
class PipelineState:
    """Mutable state during pipeline"""
    current_iteration: int = 0
    total_graphs_generated: int = 0
    total_evaluations_done: int = 0
    training_dataset_size: int = 0
    iteration_history: list = None  # Will be initialized to []
    
    def __post_init__(self):
        if self.iteration_history is None:
            self.iteration_history = []


def run_pipeline(config: Config, logger: logging.Logger) -> None:
    """
    Main optimization loop
    
    Args:
        config: Configuration object
        logger: Logger object
    """
    
    from gnn_models.model_manager import initialize_gnn_models
    from data_management.dataset_manager import load_training_dataset, save_training_dataset
    from evaluation.math_solver import load_math_problems
    from data_management.graph_storage import load_good_graphs_set
    
    logger.info("Initializing pipeline components...")

    state = PipelineState()
    
    models = initialize_gnn_models(config, logger)
    
    training_dataset = load_training_dataset(config.data_dir, logger)
    state.training_dataset_size = training_dataset.size()
    
    math_problems = load_math_problems(config, logger)
    
    good_graphs_set = load_good_graphs_set(config.data_dir, config.good_graphs_max_size, logger)
    
    logger.info(f"{'='*60}")
    logger.info("Configuration:")
    logger.info(f"  - Graphs per iteration: {config.num_graphs_per_iteration:,}")
    logger.info(f"  - Select top-K: {config.top_k_to_keep}")
    logger.info(f"  - Evaluate best: {config.eval_k_best}")
    logger.info(f"  - Max iterations: {config.max_iterations}")
    logger.info(f"  - Retrain frequency: {config.retrain_frequency}")
    logger.info(f"{'='*60}\n")
    
    for iteration_num in range(config.max_iterations):
        state.current_iteration = iteration_num
        iteration_start = time.time()
        
        logger.info(f"{'='*60}")
        logger.info(f"ITERATION {iteration_num + 1}/{config.max_iterations}")
        logger.info(f"{'='*60}")
        
        try:
            metrics = run_single_iteration(
                iteration_num, config, logger, state, 
                models, good_graphs_set, training_dataset, math_problems
            )
            
            logger.info(f"\nIteration Results:")
            logger.info(f"  - Graphs generated: {metrics.graphs_generated:,}")
            logger.info(f"  - Graphs selected: {metrics.graphs_selected}")
            logger.info(f"  - Graphs evaluated: {metrics.graphs_evaluated}")
            logger.info(f"  - Training samples added: {metrics.training_samples_added}")
            logger.info(f"  - Best predicted score: {metrics.best_predicted_score:.4f}")
            logger.info(f"  - Best actual score: {metrics.best_actual_score:.4f}")
            if metrics.gnn_retrained:
                logger.info(f"  - GNN RETRAINED (loss: {metrics.retrain_loss:.4f})")
            
            iteration_time = time.time() - iteration_start
            logger.info(f"  - Iteration time: {iteration_time:.2f}s")
            
            # Store metrics
            state.iteration_history.append(metrics)
            
            # #TODO Check if you want this!
            # # Checkpoint
            # if (iteration_num + 1) % config.checkpoint_frequency == 0:
            #     logger.info(f"\nSaving checkpoint at iteration {iteration_num + 1}...")
            #     save_checkpoint(config, logger, state, iteration_num, training_dataset, models)
            
            # # Stop conditiongit 
            # if should_stop(config, logger, state):
            #     logger.info("Stopping criteria met")
            #     break
                
        except Exception as e:
            logger.error(f"Error in iteration {iteration_num}: {e}", exc_info=True)
            raise
    
    logger.info(f"\n{'='*60}")
    logger.info("PIPELINE COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total iterations: {len(state.iteration_history)}")
    logger.info(f"Total graphs generated: {state.total_graphs_generated:,}")
    logger.info(f"Total evaluations done: {state.total_evaluations_done}")
    logger.info(f"Final training dataset size: {state.training_dataset_size}")
    
    # Save final training dataset
    save_training_dataset(training_dataset, config.data_dir, logger)

from data_management.graph_storage import GraphSet
from data_management.dataset_manager import TrainingDataset

def run_single_iteration(
    iteration_num: int,
    config: Config,
    logger: logging.Logger,
    state: PipelineState,
    models: Dict[str, Any],
    good_graphs_set: GraphSet,
    training_dataset: TrainingDataset,
    math_problems: List[Dict[str, Any]]
) -> IterationMetrics:
    """
    Execute single iteration
    
    Steps:
    1. Generate N graphs
    2. Predict performance
    3. Select best graphs
    4. Evaluate selected graphs
    5. Update training data
    6. Possibly retrain GNN
    """

    from evaluation.llm_evaluator import evaluate_selected_graphs
    
    logger.info(f"\n[Step 1/6] Generating {config.num_graphs_per_iteration:,} graphs...")
    generated_graphs = generate_graph_batch(config, logger)
    logger.info(f"  ✓ Generated {len(generated_graphs)} graphs")
    state.total_graphs_generated += len(generated_graphs)
    
    logger.info(f"\n[Step 2/6] Running GNN predictions on all graphs...")
    predictions = predict_batch_performance(config, logger, generated_graphs)
    best_predicted = max(p['score'] for p in predictions)
    logger.info(f"  ✓ Predictions complete")
    logger.info(f"  ✓ Best predicted score: {best_predicted:.4f}")
    
    logger.info(f"\n[Step 3/6] Selecting top {config.eval_k_best} graphs for evaluation...")
    selected_graphs = select_top_graphs(
        config, logger, predictions, good_graphs_set
    )
    logger.info(f"  ✓ Selected {len(selected_graphs)} graphs")
    logger.info(f"  ✓ Good graphs set size: {good_graphs_set.size()}")
    
    logger.info(f"\n[Step 4/6] Evaluating selected graphs with LLM...")
    evaluation_results = evaluate_selected_graphs(
        config, logger, selected_graphs, math_problems
    )
    best_actual = max(r['actual_score'] for r in evaluation_results) if evaluation_results else 0.0
    logger.info(f"  ✓ Evaluated {len(evaluation_results)} graphs")
    logger.info(f"  ✓ Best actual score: {best_actual:.4f}")
    state.total_evaluations_done += len(evaluation_results)
    
    logger.info(f"\n[Step 5/6] Updating training dataset...")
    num_samples_added = update_training_data(config, logger, evaluation_results, training_dataset)
    state.training_dataset_size += num_samples_added
    logger.info(f"  ✓ Added {num_samples_added} samples to training data")
    
    gnn_retrained = False
    retrain_loss = None
    
    if (iteration_num + 1) % config.retrain_frequency == 0:
        logger.info(f"\n[Step 6/6] Retraining GNN models...")
        retrain_loss = retrain_gnn_models(config, logger, state, training_dataset)
        gnn_retrained = True
        logger.info(f"  ✓ GNN retrained")
        logger.info(f"  ✓ Retrain loss: {retrain_loss:.4f}")
    else:
        logger.info(f"\n[Step 6/6] Skipping retrain (next in {config.retrain_frequency - (iteration_num + 1) % config.retrain_frequency} iterations)")
    
    # Create metrics
    metrics = IterationMetrics(
        iteration_num=iteration_num,
        timestamp=datetime.now(),
        graphs_generated=len(generated_graphs),
        graphs_selected=len(selected_graphs),
        graphs_evaluated=len(evaluation_results),
        training_samples_added=num_samples_added,
        best_predicted_score=best_predicted,
        best_actual_score=best_actual,
        gnn_retrained=gnn_retrained,
        retrain_loss=retrain_loss,
    )
    
    return metrics

# ============================================================================
# PLACEHOLDER FUNCTIONS (to be implemented in respective modules)
#TODO Clean up all functions
# ============================================================================
def generate_graph_batch(config: Config, logger: logging.Logger) -> list:
    """Generate N graphs using LangGraph templates"""
    from graph_generation.langgraph_generator import generate_langgraph_variants
    
    graphs = generate_langgraph_variants(
        num_graphs=config.num_graphs_per_iteration,
        config=config,
        logger=logger
    )
    return graphs

def predict_batch_performance(
    config: Config,
    logger: logging.Logger,
    graphs: list
) -> list:
    """Run GNN inference on all graphs"""
    from graph_conversion.serialization import serialize_langgraph_batch
    from graph_conversion.gnn_conversion import convert_to_gnn_format_batch
    
    # Serialize
    serialized = serialize_langgraph_batch(graphs, logger)
    
    # Convert to GNN
    gnn_graphs = convert_to_gnn_format_batch(serialized, logger)
    
    # TODO: Run actual GNN inference
    # For now, return dummy predictions
    predictions = []
    for gnn_graph in gnn_graphs:
        predictions.append({
            'graph': gnn_graph,
            'score': np.random.random(),  # Dummy score
        })
    
    return predictions

def update_training_data(
    config: Config,
    logger: logging.Logger,
    evaluation_results: list,
    training_dataset: TrainingDataset
) -> int:
    """Add evaluation results to training dataset"""
    from data_management.dataset_manager import add_samples_to_dataset, save_training_dataset
    
    # Add samples to dataset
    num_added = add_samples_to_dataset(
        training_dataset,
        evaluation_results,
        logger
    )

    # Save updated dataset
    save_training_dataset(
        training_dataset,
        config.data_dir,
        logger
    )   
    
    return num_added

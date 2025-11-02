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
    from data_management.graph_storage import GraphSet
    
    # Initialize components
    logger.info("Initializing pipeline components...")
    
    # Initialize GNN models
    models = initialize_gnn_models(config, logger)
    
    # Load training dataset
    training_dataset = load_training_dataset(config.data_dir, logger)
    
    # Load math problems
    math_problems = load_math_problems(config, logger)
    
    # Initialize state
    state = PipelineState()
    state.good_graph_set = GraphSet(max_size=config.good_graphs_max_size)
    
    logger.info(f"{'='*60}")
    logger.info("Configuration:")
    logger.info(f"  - Graphs per iteration: {config.num_graphs_per_iteration:,}")
    logger.info(f"  - Select top-K: {config.top_k_to_keep}")
    logger.info(f"  - Evaluate best: {config.eval_k_best}")
    logger.info(f"  - Max iterations: {config.max_iterations}")
    logger.info(f"  - Retrain frequency: {config.retrain_frequency}")
    logger.info(f"{'='*60}\n")
    
    # Main loop
    for iteration_num in range(config.max_iterations):
        state.current_iteration = iteration_num
        iteration_start = time.time()
        
        logger.info(f"{'='*60}")
        logger.info(f"ITERATION {iteration_num + 1}/{config.max_iterations}")
        logger.info(f"{'='*60}")
        
        try:
            # Run single iteration
            metrics = run_single_iteration(
                iteration_num, config, logger, state, 
                models, training_dataset, math_problems
            )
            
            # Log metrics
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
            
            # # Stop condition
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
    
    # Save final state
    save_training_dataset(training_dataset, config.data_dir, logger)

# def run_pipeline(config: Config, logger: logging.Logger) -> None:
#     """
#     Main optimization loop
    
#     Args:
#         config: Configuration object (passed throughout)
#         logger: Logger object (passed throughout)
    
#     Flow for each iteration:
#     1. Generate N=100,000 graphs
#     2. Predict performance with GNN
#     3. Select top K graphs
#     4. Evaluate selected graphs with LLM
#     5. Add results to training data
#     6. Every K iterations: retrain GNN
#     """
    
#     # Initialize state
#     state = PipelineState()
    
#     logger.info(f"Configuration:")
#     logger.info(f"  - Graphs per iteration: {config.num_graphs_per_iteration:,}")
#     logger.info(f"  - Select top-K: {config.top_k_to_keep}")
#     logger.info(f"  - Evaluate best: {config.eval_k_best}")
#     logger.info(f"  - Max iterations: {config.max_iterations}")
#     logger.info(f"  - Retrain frequency: {config.retrain_frequency}")
    
#     # Main loop
#     for iteration_num in range(config.max_iterations):
#         state.current_iteration = iteration_num
#         iteration_start = time.time()
        
#         logger.info(f"\n{'='*60}")
#         logger.info(f"ITERATION {iteration_num + 1}/{config.max_iterations}")
#         logger.info(f"{'='*60}")
        
#         try:
#             # Run single iteration
#             metrics = run_single_iteration(iteration_num, config, logger, state)
            
#             # Log metrics
#             logger.info(f"\nIteration Results:")
#             logger.info(f"  - Graphs generated: {metrics.graphs_generated:,}")
#             logger.info(f"  - Graphs selected: {metrics.graphs_selected}")
#             logger.info(f"  - Graphs evaluated: {metrics.graphs_evaluated}")
#             logger.info(f"  - Training samples added: {metrics.training_samples_added}")
#             logger.info(f"  - Best predicted score: {metrics.best_predicted_score:.4f}")
#             logger.info(f"  - Best actual score: {metrics.best_actual_score:.4f}")
#             if metrics.gnn_retrained:
#                 logger.info(f"  - GNN RETRAINED (loss: {metrics.retrain_loss:.4f})")
            
#             iteration_time = time.time() - iteration_start
#             logger.info(f"  - Iteration time: {iteration_time:.2f}s")
            
#             # Store metrics
#             state.iteration_history.append(metrics)
            
#             # Checkpoint
#             if (iteration_num + 1) % config.checkpoint_frequency == 0:
#                 logger.info(f"\nSaving checkpoint at iteration {iteration_num + 1}...")
#                 save_checkpoint(config, logger, state, iteration_num)
            
#             # Stop condition (can be improved)
#             if should_stop(config, logger, state):
#                 logger.info("Stopping criteria met")
#                 break
                
#         except Exception as e:
#             logger.error(f"Error in iteration {iteration_num}: {e}", exc_info=True)
#             raise
    
#     logger.info(f"\n{'='*60}")
#     logger.info("PIPELINE COMPLETE")
#     logger.info(f"{'='*60}")
#     logger.info(f"Total iterations: {len(state.iteration_history)}")
#     logger.info(f"Total graphs generated: {state.total_graphs_generated:,}")
#     logger.info(f"Total evaluations done: {state.total_evaluations_done}")
#     logger.info(f"Final training dataset size: {state.training_dataset_size}")

def run_single_iteration(
    iteration_num: int,
    config: Config,
    logger: logging.Logger,
    state: PipelineState,
    models: Dict[str, Any],
    training_dataset: Any,
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
    
    # Step 1: Generate graphs
    logger.info(f"\n[Step 1/6] Generating {config.num_graphs_per_iteration:,} graphs...")
    generated_graphs = generate_graph_batch(config, logger)
    logger.info(f"  ✓ Generated {len(generated_graphs)} graphs")
    state.total_graphs_generated += len(generated_graphs)
    
    # Step 2: Predict with GNN
    logger.info(f"\n[Step 2/6] Running GNN predictions on all graphs...")
    predictions = predict_batch_performance(config, logger, generated_graphs)
    best_predicted = max(p['score'] for p in predictions)
    logger.info(f"  ✓ Predictions complete")
    logger.info(f"  ✓ Best predicted score: {best_predicted:.4f}")
    
    # Step 3: Select best graphs
    logger.info(f"\n[Step 3/6] Selecting top {config.eval_k_best} graphs for evaluation...")
    selected_graphs = select_top_graphs(
        config, logger, state, predictions
    )
    logger.info(f"  ✓ Selected {len(selected_graphs)} graphs")
    logger.info(f"  ✓ Good graphs set size: {state.good_graph_set.size()}")
    
    # Step 4: Evaluate selected graphs
    logger.info(f"\n[Step 4/6] Evaluating selected graphs with LLM...")
    evaluation_results = evaluate_selected_graphs(
        config, logger, selected_graphs, math_problems
    )
    best_actual = max(r['actual_score'] for r in evaluation_results) if evaluation_results else 0.0
    logger.info(f"  ✓ Evaluated {len(evaluation_results)} graphs")
    logger.info(f"  ✓ Best actual score: {best_actual:.4f}")
    state.total_evaluations_done += len(evaluation_results)
    
    # Step 5: Update training data
    logger.info(f"\n[Step 5/6] Updating training dataset...")
    num_samples_added = update_training_data(config, logger, evaluation_results)
    state.training_dataset_size += num_samples_added
    logger.info(f"  ✓ Added {num_samples_added} samples to training data")
    
    # Step 6: Check if should retrain
    gnn_retrained = False
    retrain_loss = None
    
    if (iteration_num + 1) % config.retrain_frequency == 0:
        logger.info(f"\n[Step 6/6] Retraining GNN models...")
        retrain_loss = retrain_gnn_models(config, logger, state)
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

def select_top_graphs(
    config: Config,
    logger: logging.Logger,
    state: PipelineState,
    predictions: list
) -> list:
    """Select top-K graphs for evaluation"""
    from data_management.graph_storage import add_to_graphs_set, select_for_evaluation
    
    # Sort by score
    sorted_preds = sorted(predictions, key=lambda x: x['score'], reverse=True)
    
    # Add top K to good_graphs_set
    top_k_graphs = sorted_preds[:config.top_k_to_keep]

    # Add top K graphs to good_graphs_set
    add_to_graphs_set(
        state.good_graph_set,
        top_k_graphs,
        logger
    )
    
    # Select eval_k_best for evaluation
    selected = select_for_evaluation(
        state.good_graph_set,
        config.eval_k_best,
        logger
    )
    
    return selected

def update_training_data(
    config: Config,
    logger: logging.Logger,
    evaluation_results: list
) -> int:
    """Add evaluation results to training dataset"""
    from data_management.dataset_manager import add_samples_to_dataset, TrainingDataset
    
    if not hasattr(update_training_data, 'dataset'):
        update_training_data.dataset = TrainingDataset()
    
    num_added = add_samples_to_dataset(
        update_training_data.dataset,
        evaluation_results,
        logger,
        max_size=50000
    )
    
    return num_added

def evaluate_selected_graphs(
    config: Config,
    logger: logging.Logger,
    selected_graphs: list,
    math_problems: list
) -> list:
    """Evaluate selected graphs with LLM"""
    from evaluation.llm_evaluator import evaluate_selected_graphs as llm_eval
    
    results = llm_eval(config, logger, selected_graphs, math_problems)
    return results


# ============================================================================

# def generate_graph_batch(config: Config, logger: logging.Logger) -> list:
#     """
#     Generate N graphs using LangGraph templates
    
#     Returns:
#         List of LangGraph objects
#     """
#     logger.debug("generate_graph_batch called")
#     # TODO: Implement in graph_generation/langgraph_generator.py
#     # from graph_generation.langgraph_generator import generate_langgraph_variants
#     # return generate_langgraph_variants(config.num_graphs_per_iteration, config)
#     return []

# def predict_batch_performance(
#     config: Config,
#     logger: logging.Logger,
#     graphs: list
# ) -> list:
#     """
#     Run GNN inference on all graphs
    
#     Returns:
#         List of {'graph': ..., 'score': float}
#     """
#     logger.debug(f"predict_batch_performance called with {len(graphs)} graphs")
#     # TODO: Implement in gnn_models/model_manager.py
#     # from graph_conversion.serialization import serialize_langgraph_batch
#     # from graph_conversion.gnn_conversion import convert_to_gnn_format_batch
#     # from gnn_models.model_manager import run_gnn_inference_batch
#     # 
#     # serialized = serialize_langgraph_batch(graphs)
#     # gnn_graphs = convert_to_gnn_format_batch(serialized)
#     # predictions = run_gnn_inference_batch(config, logger, gnn_graphs)
#     # return predictions
#     return []

# def select_top_graphs(
#     config: Config,
#     logger: logging.Logger,
#     state: PipelineState,
#     predictions: list
# ) -> list:
#     """
#     Select top-K graphs for evaluation
    
#     - Add top K to good_graphs_set
#     - Select eval_k_best from good_graphs_set
    
#     Returns:
#         List of selected graphs for evaluation
#     """
#     logger.debug(f"select_top_graphs called with {len(predictions)} predictions")
#     # TODO: Implement in pipeline/selection_manager.py
#     # from pipeline.selection_manager import select_for_evaluation
#     # 
#     # sorted_preds = sorted(predictions, key=lambda x: x['score'], reverse=True)
#     # top_k = sorted_preds[:config.top_k_to_keep]
#     # state.good_graphs_set.extend([p['graph'] for p in top_k])
#     # 
#     # selected = select_for_evaluation(state.good_graphs_set, config.eval_k_best, logger)
#     # return selected
#     return []

# def evaluate_selected_graphs(
#     config: Config,
#     logger: logging.Logger,
#     selected_graphs: list
# ) -> list:
#     """
#     Evaluate selected graphs with actual LLM
    
#     Returns:
#         List of {'graph': ..., 'actual_score': float, ...}
#     """
#     logger.debug(f"evaluate_selected_graphs called with {len(selected_graphs)} graphs")
#     # TODO: Implement in evaluation/llm_evaluator.py
#     # from evaluation.llm_evaluator import evaluate_multiagent_systems
#     # results = evaluate_multiagent_systems(config, logger, selected_graphs)
#     # return results
#     return []

# def update_training_data(
#     config: Config,
#     logger: logging.Logger,
#     evaluation_results: list
# ) -> int:
#     """
#     Add evaluation results to training dataset
    
#     Returns:
#         Number of samples added
#     """
#     logger.debug(f"update_training_data called with {len(evaluation_results)} results")
#     # TODO: Implement in data_management/dataset_manager.py
#     # from data_management.dataset_manager import add_samples_to_dataset
#     # num_added = add_samples_to_dataset(config, logger, evaluation_results)
#     # return num_added
#     return 0

# def retrain_gnn_models(
#     config: Config,
#     logger: logging.Logger,
#     state: PipelineState
# ) -> float:
#     """
#     Retrain GNN models on updated training data
    
#     Returns:
#         Loss value from retraining
#     """
#     logger.debug("retrain_gnn_models called")
#     # TODO: Implement in gnn_models/model_manager.py
#     # from gnn_models.model_manager import retrain_gnn_models
#     # loss = retrain_gnn_models(config, logger)
#     # return loss
#     return 0.0

# def save_checkpoint(
#     config: Config,
#     logger: logging.Logger,
#     state: PipelineState,
#     iteration_num: int
# ) -> None:
#     """Save checkpoint of current state"""
#     logger.debug(f"save_checkpoint called at iteration {iteration_num}")
#     # TODO: Implement
#     pass

# def should_stop(
#     config: Config,
#     logger: logging.Logger,
#     state: PipelineState
# ) -> bool:
#     """Check if should stop pipeline"""
#     # Can add convergence criteria here
#     return False

import logging
import time
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from config.settings import Config
import numpy as np
from typing import Any, Dict, List
from data_management.graph_storage import GraphSet

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
    from gnn_models.model_manager import initialize_gnn_model, retrain_gnn_model
    from evaluation.math_solver import load_math_problems
    from data_management.graph_storage import load_good_graphs_set, load_training_dataset, save_training_dataset
    from post_processing.metrics_aggregator import create_metrics_dataframe
    
    logger.info("Initializing pipeline components...")    
    state = PipelineState()
    
    model = initialize_gnn_model(config, logger)
    
    training_dataset = load_training_dataset(config.data_dir, logger) # This is GraphSet object that stores training data
    state.training_dataset_size = training_dataset.size()

    if state.training_dataset_size > 0:
        logger.info(f"{'='*60}")
        logger.info(f"Initial training dataset exists. Training the initial model...")
        _, model = retrain_gnn_model(config, logger, model, training_dataset)
        logger.info(f"Model retrained on existing training data ({state.training_dataset_size} samples).")
        logger.info(f"{'='*60}")
    
    math_problems = load_math_problems(config, logger)
    
    good_graphs_set = load_good_graphs_set(config.data_dir, logger) # This is a GraphSet object that stores all the potential Graph objects
    
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
                model, good_graphs_set, training_dataset, math_problems
            )
            
            iteration_time = time.time() - iteration_start
            logger.info(f"  - Iteration time: {iteration_time:.2f}s")
            
            state.iteration_history.append(metrics)
                
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
    
    save_training_dataset(training_dataset, config.data_dir, logger)

    metrics_df = create_metrics_dataframe(state.iteration_history, config, logger)
    
    return metrics_df

def run_single_iteration(
    iteration_num: int,
    config: Config,
    logger: logging.Logger,
    state: PipelineState,
    model: Dict[str, Any],
    good_graphs_set: GraphSet,
    training_dataset: GraphSet,
    math_problems: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    IDEA for this part: maybe every step should have an output of a dict metrics["step_name"] = { ... },
    at the end of the look, we also call function evaluate_loop and we get the loop
    metrics metrics["loop"] = { ... } and we merge them all in the metrics dataclass

    For this, we will have to change the dataclass and the return type accordingly.
    """

    from graph_generation.graph_generation import generate_graph_batch
    from gnn_models.model_manager import predict_batch_performance, retrain_gnn_model
    from evaluation.llm_evaluator import evaluate_selected_graphs
    from data_management.graph_storage import select_top_graphs, update_training_data
    
    logger.info(f"\n[Step 1/6] Generating {config.num_graphs_per_iteration:,} graphs...")
    metrics1, generated_graphs = generate_graph_batch(config, logger, training_dataset)
    logger.info(f"  ✅ Generated {generated_graphs.size()} graphs")
    
    logger.info(f"\n[Step 2/6] Running GNN predictions on all graphs...")
    metrics2, predictions = predict_batch_performance(config, logger, model, generated_graphs) # Input is list of Graph objects GraphSet.to_pyg() will be used to predict This function updates GraphSet objects to have paramets gnn_predic set
    best_predicted = metrics2['best_predicted']
    logger.info(f"  ✅ Predictions complete")
    logger.info(f"  ✅ Best predicted score: {best_predicted:.4f}")
    
    logger.info(f"\n[Step 3/6] Selecting top {config.eval_k_best} graphs for evaluation...")
    metrics3, selected_graphs = select_top_graphs(config, logger, good_graphs_set, predictions) 
    # ATTENTION: check if good_graphs_set is changing throughout the iterations if something is going wrong
    logger.info(f"  ✅ Selected {selected_graphs.size()} graphs")
    logger.info(f"  ✅ Good graphs set size: {good_graphs_set.size()}")
    
    logger.info(f"\n[Step 4/6] Evaluating selected graphs with LLM...")
    metrics4, evaluation_results = evaluate_selected_graphs(config, logger, selected_graphs, math_problems)
    best_actual = metrics4['best_evaluated']
    logger.info(f"  ✅ Evaluated {evaluation_results.size()} graphs")
    logger.info(f"  ✅ Best actual score: {best_actual:.4f}")
    
    logger.info(f"\n[Step 5/6] Updating training dataset...")
    metrics5 = update_training_data(config, logger, evaluation_results, training_dataset) #To je graphset
    logger.info(f"  ✅ Added {metrics5['num_training_samples_added']} samples to training data")
    logger.info(f"  ✅ Training dataset size: {metrics5['total_training_dataset_size']}")
    
    retrain_loss = None
    logger.info(f"\n[Step 6/6] Retraining GNN models...")
    metrics6, model = retrain_gnn_model(config, logger, model, training_dataset) # Correct this to be the right data
    logger.info(f"  ✅ GNN retrained")

    iteration_metrics = {
        'iteration_num': iteration_num,
        'timestamp': datetime.now(),
        'step_1_generation': metrics1,
        'step_2_prediction': metrics2,
        'step_3_selection': metrics3,
        'step_4_evaluation': metrics4,
        'step_5_training_update': metrics5,
        'step_6_retraining': metrics6,
        'loop': {} 
    }
    
    return iteration_metrics

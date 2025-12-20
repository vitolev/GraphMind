import logging
import random
import time
from typing import List, Dict, Any, Tuple
from config.settings import Config
from evaluation.test_llm import build_langgraph, visualize_graph_ascii, AgentState, GlobalKnowledge  
import numpy as np
from data_management.graph_storage import GraphSet

def _evaluate_answer(
    llm_output: str,
    expected_output: str,
    problem: str,
    logger: logging.Logger
) -> float:
    """
    Evaluate LLM output against expected output.
    
    Returns a score between 0.0 and 1.0:
    - 1.0: Exact match (case-insensitive)
    - 0.7: Contains expected answer
    - 0.5: Partial match
    - 0.0: No match
    """
    
    llm_output_lower = llm_output.lower().strip()
    expected_lower = expected_output.lower().strip()
    
    # Exact match
    if llm_output_lower == expected_lower:
        score = 1.0
        match_type = "EXACT_MATCH"
    
    # Contains expected answer
    elif expected_lower in llm_output_lower:
        score = 0.7
        match_type = "CONTAINS"
    
    # Expected answer appears as word boundary
    elif any(word in llm_output_lower.split() for word in expected_lower.split()):
        score = 0.5
        match_type = "PARTIAL_WORD_MATCH"
    
    # No match
    else:
        score = 0.0
        match_type = "NO_MATCH"
    
    logger.debug(
        f"  Problem: {problem[:60]}...\n"
        f"  Expected: {expected_output}\n"
        f"  LLM Output: {llm_output[:100]}...\n"
        f"  Match Type: {match_type}\n"
        f"  Score: {score:.2f}"
    )
    
    return score

def evaluate_selected_graphs(
    config: Config,
    logger: logging.Logger,
    selected_graphs: GraphSet,
    math_problems: Any,
) -> Tuple[Dict[str, Any], GraphSet]:
    
    step_start = time.time()
    num_graphs = selected_graphs.size()
    
    logger.debug(f"Starting evaluation of {num_graphs} selected graphs")
    
    scores = []
    gnn_scores = []
    
    # for graph in selected_graphs.get_all():
    #     try:
    #         num_nodes = len(graph.get_nodes())
    #         num_edges = len(graph.get_edges())
            
    #         if num_nodes > 0:
    #             structure_score = num_edges / (num_nodes ** 2)
    #         else:
    #             structure_score = 0.0
            
    #         random_component = np.random.uniform(0.1, 0.15)
    #         score = structure_score + random_component
            
    #         node_types = [node_type for _, node_type in graph.nodes]
    #         if 'Solver' in node_types:
    #             score += 0.2
            
    #         score = min(1.0, max(0.0, score))
            
    #         graph.set_llm_score(score, time=0.1)
    #         scores.append(score)

    #         gnn_scores.append(graph.get_gnn_score())
            
    #     except Exception as e:
    #         logger.warning(f"Error evaluating graph: {e}")
    #         graph.set_llm_score(0.0, time=0.1)
    #         scores.append(0.0)
    #         continue

    for graph_idx, graph in enumerate(selected_graphs.get_all()):
        try:
            logger.info(f"\n[Graph {graph_idx + 1}/{num_graphs}] Evaluating graph with {len(graph.get_nodes())} nodes")
                    # DEBUG: Check math_problems
            logger.debug(f"math_problems type: {type(math_problems)}, length: {len(math_problems) if math_problems else 0}")
            logger.debug(f"math_problems: {math_problems}")
            
            # Build LangGraph from structure
            compiled_graph = build_langgraph(graph.get_nodes(), graph.get_edges())
            
            graph_problem_scores = []
            graph_execution_times = []
            
            # Run on each problem
            for prob_idx, problem_data in enumerate(math_problems[:5]): ### Limit to first 5 problems for faster testing !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                problem = problem_data["question"]
                expected = problem_data["answer"]
                category = "hihi" #problem_data.get("category", "unknown")
                
                logger.debug(f"\n  [{prob_idx + 1}/{len(math_problems)}] Category: {category}")
                
                # Build initial state
                initial_state: AgentState = {
                    "problem": [problem],
                    "global_knowledge": GlobalKnowledge(),
                    "result": [],
                    "node_type": None,
                    "node_id": None,
                    "solution": None
                }
                
                exec_start = time.time()
                
                try:
                    result = compiled_graph.invoke(initial_state)
                    execution_time = time.time() - exec_start
                    
                    # Extract solution (now a string, not a list)
                    llm_output = result.get('solution', '')
                    
                    # Evaluate answer
                    score = _evaluate_answer(llm_output, expected, problem, logger)
                    
                except Exception as e:
                    execution_time = time.time() - exec_start
                    logger.warning(f"Error executing multi-agent system: {e}")
                    score = 0.0
                
                graph_problem_scores.append(score)
                graph_execution_times.append(execution_time)
                
                logger.debug(f"  [{prob_idx + 1}/{len(math_problems)}] → {score:.2f}\n")
            
            # Average score across all problems
            graph_llm_score = float(np.mean(graph_problem_scores)) if graph_problem_scores else 0.0
            avg_execution_time = float(np.mean(graph_execution_times)) if graph_execution_times else 0.0
            
            graph.set_llm_score(graph_llm_score, time=avg_execution_time)
            scores.append(graph_llm_score)
            gnn_scores.append(graph.get_gnn_score())
            
            logger.info(
                f"[Graph {graph_idx + 1}] "
                f"LLM Score: {graph_llm_score:.4f} | "
                f"Problem Scores: {[f'{s:.2f}' for s in graph_problem_scores]}"
            )
            
        except Exception as e:
            logger.warning(f"Error evaluating graph: {e}")
            graph.set_llm_score(0.0, time=0.0)
            scores.append(0.0)
            continue

    
    evaluation_time = time.time() - step_start
    scores_array = np.array(scores)
    scores_array = np.array(scores)
    gnn_scores_array = np.array(gnn_scores)
    # Compute RMSE only if we have all predictions and ground truth
    if len(scores_array) == len(gnn_scores_array) and len(scores_array) > 0:
        rmse = float(np.sqrt(np.mean((scores_array - gnn_scores_array) ** 2)))
    else:
        rmse = None
    

    metrics = {
        'step_name': 'llm_evaluation',
        'duration_seconds': round(evaluation_time, 4),
        'num_graphs': num_graphs,
        'num_problems': len(math_problems),
        'num_evaluations': num_graphs * len(math_problems),
        'best_evaluated': float(scores_array.max()) if len(scores_array) > 0 else None,
        'worst_evaluated': float(scores_array.min()) if len(scores_array) > 0 else None,
        'mean_evaluated': float(scores_array.mean()) if len(scores_array) > 0 else None,
        'std_evaluated': float(scores_array.std()) if len(scores_array) > 0 else None,
        'rmse_gnn_vs_llm': rmse,
        'metadata': {
            'evaluation_time_per_graph': round(evaluation_time / num_graphs, 4) if num_graphs > 0 else 0,
            'evaluation_time_per_problem': round(evaluation_time / (num_graphs * len(math_problems)), 4) if num_graphs * len(math_problems) > 0 else 0,
            'evaluation_method': 'llm_multiagent_system',
            'num_problems_per_graph': len(math_problems),
            'problem_categories': list(set([p.get("category", "unknown") for p in math_problems]))
        }
    }

    logger.debug(
        f"Evaluation complete - "
        f"Best: {metrics['best_evaluated']:.4f}, "
        f"Mean: {metrics['mean_evaluated']:.4f}, "
        f"RMSE (GNN vs LLM): {metrics['rmse_gnn_vs_llm']}, "
        f"Time: {evaluation_time:.4f}s"
    )
    
    return metrics, selected_graphs

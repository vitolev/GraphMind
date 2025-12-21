import logging
import random
import time
from typing import List, Dict, Any, Tuple
from config.settings import Config
from evaluation.graph_builder import build_langgraph, visualize_graph_ascii
from evaluation.agent_state import AgentState, GlobalKnowledge
from evaluation.llm_callers import set_llm_provider
import numpy as np
from data_management.graph_storage import GraphSet


def _calculate_deterministic_graph_score(nodes: List[Tuple[int, str]], edges: List[Tuple[int, int]]) -> float:
    """
    Calculate a deterministic score based on graph structure.
    
    Scoring rules (designed to be learnable by GNN):
    - Base score: 0.1 (minimum for any valid graph)
    - +0.3 for each Solver node
    - +0.2 bonus if Python_solver directly precedes Solver (edge exists)
    - +0.15 for each Combine_all node (indicates merging strategy)
    - +0.1 for each Validator node (quality assurance)
    - +0.05 for each Extract_topic node (problem understanding)
    - +0.1 if graph has at least one Solver at the end (connected to END or no outgoing edges)
    - -0.05 for each excessive node (>10 nodes total)
    - -0.1 if no Solver nodes exist
    
    Maximum theoretical score: ~1.5 (with optimal structure)
    Returns score in range [0.0, 1.0] (clamped)
    """
    # Build helper structures
    node_dict = {nid: ntype for nid, ntype in nodes}
    graph_out = {}
    graph_in = {}
    
    for src, dst in edges:
        if src not in graph_out:
            graph_out[src] = []
        if dst not in graph_in:
            graph_in[dst] = []
        graph_out[src].append(dst)
        graph_in[dst].append(src)
    
    score = 0.1  # Base score
    
    # Count nodes by type
    solver_count = 0
    python_solver_count = 0
    combine_all_count = 0
    validator_count = 0
    extract_topic_count = 0
    
    solver_nodes = []
    
    for node_id, node_type in nodes:
        if node_type == "Solver":
            solver_count += 1
            solver_nodes.append(node_id)
        elif node_type == "Python_solver":
            python_solver_count += 1
        elif node_type == "Combine_all":
            combine_all_count += 1
        elif node_type == "Validator":
            validator_count += 1
        elif node_type == "Extract_topic":
            extract_topic_count += 1
    
    # Score for Solver nodes
    score += solver_count * 0.3
    
    # Bonus if Python_solver directly precedes Solver
    for src, dst in edges:
        src_type = node_dict.get(src)
        dst_type = node_dict.get(dst)
        if src_type == "Python_solver" and dst_type == "Solver":
            score += 0.2  # Bonus for Python_solver -> Solver pattern
    
    # Score for other helpful nodes
    score += combine_all_count * 0.15
    score += validator_count * 0.1
    score += extract_topic_count * 0.05
    
    # Check if solver is at the end (has no outgoing edges or connects to END)
    has_end_solver = False
    for solver_id in solver_nodes:
        # Check if solver has no outgoing edges (likely end node)
        if solver_id not in graph_out or len(graph_out[solver_id]) == 0:
            has_end_solver = True
            break
        # Check if solver connects to END node
        for dst in graph_out.get(solver_id, []):
            if node_dict.get(dst) == "END":
                has_end_solver = True
                break
        if has_end_solver:
            break
    
    if has_end_solver:
        score += 0.1
    
    # Penalty for excessive nodes
    total_nodes = len([n for n in nodes if n[1] not in ["START", "END"]])
    if total_nodes > 10:
        excess = total_nodes - 10
        score -= excess * 0.05
    
    # Heavy penalty if no solvers
    if solver_count == 0:
        score -= 0.1
    
    # Clamp to [0.0, 1.0] range
    score = max(0.0, min(1.0, score))
    
    return score

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
    # Set LLM provider based on config
    set_llm_provider(
        provider=getattr(config, 'llm_provider', 'groq'),
        local_model=getattr(config, 'local_llm_model', 'microsoft/Phi-3-mini-4k-instruct'),
        local_device=getattr(config, 'local_llm_device', 'auto'),
        ollama_model=getattr(config, 'ollama_model', 'llama3.2'),
        ollama_base_url=getattr(config, 'ollama_base_url', 'http://localhost:11434')
    )
    
    step_start = time.time()
    num_graphs = selected_graphs.size()
    
    logger.debug(f"Starting evaluation of {num_graphs} selected graphs")
    
    scores = []
    gnn_scores = []
    per_graph_metrics = []  # Store detailed metrics for each graph
    
    # Check if we should simulate LLM evaluation
    simulate = getattr(config, 'simulate_llm_evaluation', False)
    
    if simulate:
        logger.info("🎯 SIMULATED LLM EVALUATION MODE: Using deterministic graph-based scoring")
    else:
        # Set LLM provider only if not simulating
        set_llm_provider(
            provider=getattr(config, 'llm_provider', 'groq'),
            local_model=getattr(config, 'local_llm_model', 'microsoft/Phi-3-mini-4k-instruct'),
            local_device=getattr(config, 'local_llm_device', 'auto'),
            ollama_model=getattr(config, 'ollama_model', 'llama3.2'),
            ollama_base_url=getattr(config, 'ollama_base_url', 'http://localhost:11434')
        )
    
    # Determine how many problems to evaluate per graph
    num_problems_to_eval = min(config.num_eval_problems, len(math_problems) if math_problems else 0)
    logger.info(f"Evaluating {num_problems_to_eval} problems per graph (from {len(math_problems) if math_problems else 0} available)")

    for graph_idx, graph in enumerate(selected_graphs.get_all()):
        try:
            logger.info(f"\n[Graph {graph_idx + 1}/{num_graphs}] Evaluating graph with {len(graph.get_nodes())} nodes")
            
            if simulate:
                # SIMULATED MODE: Use deterministic graph-based scoring
                base_score = _calculate_deterministic_graph_score(graph.get_nodes(), graph.get_edges())
                
                # Use same score for all problems (fully deterministic)
                # Same graph structure = same score every time
                graph_problem_scores = []
                graph_execution_times = []
                problem_match_types = []
                
                for prob_idx in range(num_problems_to_eval):
                    # Use the same base_score for all problems (fully deterministic)
                    problem_score = base_score
                    
                    graph_problem_scores.append(problem_score)
                    graph_execution_times.append(0.01)  # Fast simulated execution
                    
                    # Assign match type based on score
                    if problem_score >= 0.9:
                        match_type = "EXACT_MATCH"
                    elif problem_score >= 0.7:
                        match_type = "CONTAINS"
                    elif problem_score >= 0.5:
                        match_type = "PARTIAL"
                    else:
                        match_type = "NO_MATCH"
                    problem_match_types.append(match_type)
                    
                    logger.debug(f"  [{prob_idx + 1}/{num_problems_to_eval}] → {problem_score:.2f} (simulated, deterministic)")
                
                logger.info(f"  📊 Simulated score: {base_score:.4f} (deterministic, same for all {num_problems_to_eval} problems)")
            else:
                # REAL MODE: Actual LLM evaluation
                # DEBUG: Check math_problems
                logger.debug(f"math_problems type: {type(math_problems)}, length: {len(math_problems) if math_problems else 0}")
                logger.debug(f"math_problems: {math_problems}")
                
                # Build LangGraph from structure
                compiled_graph = build_langgraph(graph.get_nodes(), graph.get_edges())
                
                graph_problem_scores = []
                graph_execution_times = []
                problem_match_types = []  # Track match types for each problem
                
                # Run on each problem (use config value, not hardcoded)
                problems_to_evaluate = math_problems[:num_problems_to_eval]
                for prob_idx, problem_data in enumerate(problems_to_evaluate):
                    problem = problem_data["question"]
                    expected = problem_data["answer"]
                    category = problem_data.get("category", "unknown")
                    
                    logger.debug(f"\n  [{prob_idx + 1}/{num_problems_to_eval}] Category: {category}")
                    
                    # Build initial state
                    from evaluation.agent_state import ScopedKnowledge
                    initial_state: AgentState = {
                        "problem": [problem],
                        "global_knowledge": GlobalKnowledge(),
                        "graph_structure": None,  # Will be set by graph builder
                        "scoped_knowledge": {"root": ScopedKnowledge(scope_id="root")},
                        "scope_mapping": {},
                        "current_scope": "root",
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
                        
                        # Determine match type for tracking
                        match_type = "EXACT_MATCH" if score == 1.0 else "CONTAINS" if score == 0.7 else "PARTIAL" if score == 0.5 else "NO_MATCH"
                        problem_match_types.append(match_type)
                        
                    except Exception as e:
                        execution_time = time.time() - exec_start
                        logger.warning(f"Error executing multi-agent system: {e}")
                        score = 0.0
                        problem_match_types.append("ERROR")
                    
                    graph_problem_scores.append(score)
                    graph_execution_times.append(execution_time)
                    
                    logger.debug(f"  [{prob_idx + 1}/{num_problems_to_eval}] → {score:.2f}\n")
            
            # Calculate per-graph statistics
            graph_llm_score = float(np.mean(graph_problem_scores)) if graph_problem_scores else 0.0
            graph_llm_std = float(np.std(graph_problem_scores)) if graph_problem_scores else 0.0
            graph_llm_variance = float(np.var(graph_problem_scores)) if graph_problem_scores else 0.0
            avg_execution_time = float(np.mean(graph_execution_times)) if graph_execution_times else 0.0
            graph_gnn_score = graph.get_gnn_score()
            
            # Count problem outcomes
            num_perfect = sum(1 for s in graph_problem_scores if s == 1.0)
            num_partial = sum(1 for s in graph_problem_scores if 0.0 < s < 1.0)
            num_failed = sum(1 for s in graph_problem_scores if s == 0.0)
            
            # Calculate prediction error
            gnn_llm_error = abs(graph_gnn_score - graph_llm_score)
            
            graph.set_llm_score(graph_llm_score, time=avg_execution_time)
            scores.append(graph_llm_score)
            gnn_scores.append(graph_gnn_score)
            
            # Store detailed per-graph metrics
            per_graph_metrics.append({
                'gnn_predicted_score': graph_gnn_score,
                'llm_average_score': graph_llm_score,
                'llm_std': graph_llm_std,
                'llm_variance': graph_llm_variance,
                'gnn_llm_absolute_error': gnn_llm_error,
                'num_problems_evaluated': len(graph_problem_scores),
                'num_perfect_scores': num_perfect,
                'num_partial_scores': num_partial,
                'num_failed_scores': num_failed,
                'problem_scores': graph_problem_scores,  # For visualization
                'avg_execution_time': avg_execution_time
            })
            
            logger.info(
                f"[Graph {graph_idx + 1}] "
                f"GNN Predicted: {graph_gnn_score:.4f} | "
                f"LLM Average: {graph_llm_score:.4f} | "
                f"Error: {gnn_llm_error:.4f} | "
                f"Perfect: {num_perfect}/{len(graph_problem_scores)} | "
                f"Std: {graph_llm_std:.4f}"
            )
            
        except Exception as e:
            logger.warning(f"Error evaluating graph {graph_idx + 1}: {e}")
            graph.set_llm_score(0.0, time=0.0)
            scores.append(0.0)
            gnn_scores.append(graph.get_gnn_score())
            # Add error entry to per_graph_metrics
            per_graph_metrics.append({
                'gnn_predicted_score': graph.get_gnn_score(),
                'llm_average_score': 0.0,
                'llm_std': 0.0,
                'llm_variance': 0.0,
                'gnn_llm_absolute_error': graph.get_gnn_score(),
                'num_problems_evaluated': 0,
                'num_perfect_scores': 0,
                'num_partial_scores': 0,
                'num_failed_scores': 0,
                'problem_scores': [],
                'avg_execution_time': 0.0,
                'error': str(e)
            })
            continue

    
    evaluation_time = time.time() - step_start
    scores_array = np.array(scores)
    gnn_scores_array = np.array(gnn_scores)
    
    # Compute aggregate prediction metrics
    if len(scores_array) == len(gnn_scores_array) and len(scores_array) > 0:
        rmse = float(np.sqrt(np.mean((scores_array - gnn_scores_array) ** 2)))
        mae = float(np.mean(np.abs(scores_array - gnn_scores_array)))
        
        # Correlation coefficient
        if len(scores_array) > 1:
            correlation = float(np.corrcoef(scores_array, gnn_scores_array)[0, 1])
        else:
            correlation = None
    else:
        rmse = None
        mae = None
        correlation = None
    
    # Aggregate variance metrics (mean variance across all graphs)
    graph_variances = [m['llm_variance'] for m in per_graph_metrics if 'llm_variance' in m]
    mean_variance_across_graphs = float(np.mean(graph_variances)) if graph_variances else None
    mean_std_across_graphs = float(np.mean([m['llm_std'] for m in per_graph_metrics if 'llm_std' in m])) if per_graph_metrics else None
    
    # Aggregate success metrics
    total_perfect = sum(m.get('num_perfect_scores', 0) for m in per_graph_metrics)
    total_partial = sum(m.get('num_partial_scores', 0) for m in per_graph_metrics)
    total_failed = sum(m.get('num_failed_scores', 0) for m in per_graph_metrics)
    total_problems_evaluated = sum(m.get('num_problems_evaluated', 0) for m in per_graph_metrics)
    
    # Calculate actual number of evaluations (not theoretical)
    actual_num_evaluations = total_problems_evaluated

    metrics = {
        'step_name': 'llm_evaluation',
        'duration_seconds': round(evaluation_time, 4),
        'num_graphs': num_graphs,
        'num_problems_available': len(math_problems) if math_problems else 0,
        'num_problems_per_graph': num_problems_to_eval,
        'num_evaluations': actual_num_evaluations,  # Actual, not theoretical
        'best_evaluated': float(scores_array.max()) if len(scores_array) > 0 else None,
        'worst_evaluated': float(scores_array.min()) if len(scores_array) > 0 else None,
        'mean_evaluated': float(scores_array.mean()) if len(scores_array) > 0 else None,
        'std_evaluated': float(scores_array.std()) if len(scores_array) > 0 else None,
        
        # GNN vs LLM prediction metrics
        'rmse_gnn_vs_llm': rmse,
        'mae_gnn_vs_llm': mae,
        'correlation_gnn_llm': correlation,
        'mean_absolute_error_per_graph': mae,  # Same as mae, for consistency
        
        # Variance/consistency metrics
        'mean_variance_across_graphs': mean_variance_across_graphs,
        'mean_std_across_graphs': mean_std_across_graphs,
        
        # Success rate metrics
        'total_perfect_scores': total_perfect,
        'total_partial_scores': total_partial,
        'total_failed_scores': total_failed,
        'success_rate_perfect': total_perfect / total_problems_evaluated if total_problems_evaluated > 0 else 0.0,
        'success_rate_any': (total_perfect + total_partial) / total_problems_evaluated if total_problems_evaluated > 0 else 0.0,
        
        # Per-graph details (for detailed analysis/visualization)
        'per_graph_metrics': per_graph_metrics,
        
        'metadata': {
            'evaluation_time_per_graph': round(evaluation_time / num_graphs, 4) if num_graphs > 0 else 0,
            'evaluation_time_per_problem': round(evaluation_time / actual_num_evaluations, 4) if actual_num_evaluations > 0 else 0,
            'evaluation_method': 'llm_multiagent_system',
            'problem_categories': list(set([p.get("category", "unknown") for p in math_problems])) if math_problems else []
        }
    }

    # Format optional metrics safely
    rmse_str = f"{metrics['rmse_gnn_vs_llm']:.4f}" if metrics['rmse_gnn_vs_llm'] is not None else "N/A"
    mae_str = f"{metrics['mae_gnn_vs_llm']:.4f}" if metrics['mae_gnn_vs_llm'] is not None else "N/A"
    corr_str = f"{metrics['correlation_gnn_llm']:.4f}" if metrics['correlation_gnn_llm'] is not None else "N/A"
    
    logger.info(
        f"Evaluation complete - "
        f"Graphs: {num_graphs}, Problems/graph: {num_problems_to_eval}, "
        f"Best LLM: {metrics['best_evaluated']:.4f}, Mean LLM: {metrics['mean_evaluated']:.4f}, "
        f"RMSE (GNN vs LLM): {rmse_str}, "
        f"MAE: {mae_str}, "
        f"Correlation: {corr_str}, "
        f"Perfect: {total_perfect}/{total_problems_evaluated}, "
        f"Time: {evaluation_time:.4f}s"
    )
    
    return metrics, selected_graphs

"""
#TODO Real data loading: not synthetic problems

Load and manage math problems dataset

This module handles loading math problems that will be used
to evaluate multiagent systems.
"""

import logging
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from datasets import load_dataset

def load_math_problems(
    config,
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    Load math problems dataset
    
    Args:
        config: Configuration object with num_eval_problems, problem_difficulties
        logger: Logger
    
    Returns:
        List of math problem dictionaries
        Each problem:
        {
            'id': str,
            'question': str,
            'answer': str,
        }
    """
    
    logger.info(f"Loading {config.num_eval_problems} math problems...")
    
    # Set seed for reproducibility
    np.random.seed(config.seed)

    start_int = np.random.randint(0, 10000 - config.num_eval_problems)
    problems = load_dataset("nvidia/OpenMathInstruct-1", split=f"train[{start_int}:{start_int + config.num_eval_problems}]")

    df = problems.to_pandas()
    df = df[['question', 'expected_answer']]
    df = df.rename(columns={'expected_answer': 'answer'})
    df['id'] = df.index + start_int

    # Convert to list of dicts
    problems = df.to_dict(orient='records')
    
    logger.info(f"  ✓ Loaded {len(problems)} math problems")
    
    return problems

def _create_synthetic_problems(
    num_problems: int,
    difficulties: List[str],
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    Create synthetic math problems for testing
    
    In production, this would load from a real dataset
    (e.g., MATH dataset, Competition Math, etc.)
    
    Args:
        num_problems: Number of problems to create
        difficulties: List of difficulty levels
        logger: Logger
    
    Returns:
        List of problem dictionaries
    """
    
    import random
    
    problems = []
    
    # Distribute problems across difficulties
    per_difficulty = num_problems // len(difficulties)
    
    for i, difficulty in enumerate(difficulties):
        for j in range(per_difficulty):
            problem_id = i * per_difficulty + j
            
            # Create synthetic problem based on difficulty
            if difficulty == 'easy':
                problem = {
                    'id': f'problem_easy_{problem_id}',
                    'question': f'What is 5 + 3?',
                    'answer': '8',
                    'difficulty': 'easy',
                }
            elif difficulty == 'medium':
                problem = {
                    'id': f'problem_medium_{problem_id}',
                    'question': f'Solve: x^2 - 5x + 6 = 0',
                    'answer': 'x = 2 or x = 3',
                    'difficulty': 'medium',
                }
            else:  # hard
                problem = {
                    'id': f'problem_hard_{problem_id}',
                    'question': f'Find the limit of (x^2 - 1)/(x - 1) as x approaches 1',
                    'answer': '2',
                    'difficulty': 'hard',
                }
            
            problems.append(problem)
    
    logger.debug(f"Created {len(problems)} synthetic math problems")
    return problems

"""
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

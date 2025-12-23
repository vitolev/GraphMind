"""
Main script to run full distribution analysis.

This script runs both parts sequentially:
1. Generate distribution research data
2. Fit distributions to the data
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import logging
import subprocess

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run both parts of the distribution analysis."""
    logger.info("=" * 60)
    logger.info("RUNNING FULL DISTRIBUTION ANALYSIS")
    logger.info("=" * 60)
    
    # Part 1: Generate data
    logger.info("\n" + "=" * 60)
    logger.info("PART 1: Generating Distribution Research Data")
    logger.info("=" * 60)
    
    script1 = Path(__file__).parent / "generate_distribution_data.py"
    result1 = subprocess.run([sys.executable, str(script1)], capture_output=False)
    
    if result1.returncode != 0:
        logger.error("Part 1 failed. Aborting.")
        return
    
    # Part 2: Fit distributions
    logger.info("\n" + "=" * 60)
    logger.info("PART 2: Fitting Distributions")
    logger.info("=" * 60)
    
    script2 = Path(__file__).parent / "fit_distributions.py"
    result2 = subprocess.run([sys.executable, str(script2)], capture_output=False)
    
    if result2.returncode != 0:
        logger.error("Part 2 failed.")
        return
    
    logger.info("\n" + "=" * 60)
    logger.info("FULL ANALYSIS COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()




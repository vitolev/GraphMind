import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

class TrainingDataset:
    """In-memory training dataset"""
    
    def __init__(self):
        self.samples: List[Dict[str, Any]] = []
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
    
    def add_samples(self, new_samples: List[Dict[str, Any]]) -> int:
        """
        Add samples to dataset
        
        Args:
            new_samples: List of {'graph': ..., 'actual_score': ...}
        
        Returns:
            Number of samples added
        """
        self.samples.extend(new_samples)
        self.last_updated = datetime.now()
        return len(new_samples)
    
    def get_samples(self) -> List[Dict[str, Any]]:
        """Get all samples"""
        return self.samples
    
    def size(self) -> int:
        """Get number of samples"""
        return len(self.samples)
    
    def split_train_val(self, val_ratio: float = 0.2) -> Tuple[List, List]:
        """
        Split into train/val sets
        
        Args:
            val_ratio: Fraction to use for validation
        
        Returns:
            (train_samples, val_samples)
        """
        split_idx = int(len(self.samples) * (1 - val_ratio))
        train = self.samples[:split_idx]
        val = self.samples[split_idx:]
        return train, val
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'samples': self.samples,
            'size': len(self.samples),
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
        }

def load_training_dataset(
    config_data_dir: Path,
    logger: logging.Logger
) -> TrainingDataset:
    """
    Load training dataset from disk
    
    If file doesn't exist, returns empty dataset
    
    Args:
        config_data_dir: Path to data directory
        logger: Logger
    
    Returns:
        TrainingDataset object
    """
    dataset = TrainingDataset()
    dataset_path = config_data_dir / "training_dataset.pkl"
    
    if dataset_path.exists():
        try:
            with open(dataset_path, 'rb') as f:
                loaded = pickle.load(f)
            dataset.samples = loaded.get('samples', [])
            logger.info(f"Loaded training dataset with {dataset.size()} samples from {dataset_path}")
        except Exception as e:
            logger.warning(f"Could not load training dataset: {e}, starting fresh")
    else:
        logger.info(f"No existing training dataset found at {dataset_path}, starting fresh")
    
    return dataset

def save_training_dataset(
    dataset: TrainingDataset,
    config_data_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Save training dataset to disk
    
    Args:
        dataset: TrainingDataset to save
        config_data_dir: Path to data directory
        logger: Logger
    """
    config_data_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = config_data_dir / "training_dataset.pkl"
    
    try:
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset.to_dict(), f)
        logger.debug(f"Saved training dataset ({dataset.size()} samples) to {dataset_path}")
    except Exception as e:
        logger.error(f"Failed to save training dataset: {e}")
        raise

def add_samples_to_dataset(
    dataset: TrainingDataset,
    evaluation_results: List[Dict[str, Any]],
    logger: logging.Logger,
) -> int:
    """
    Add new samples to dataset and manage size
    
    Args:
        dataset: TrainingDataset object
        evaluation_results: New results from evaluation
                           List of {'graph': ..., 'actual_score': ...}
        logger: Logger
    
    Returns:
        Number of samples actually added
    """
    
    # Validate input
    if not evaluation_results:
        return 0
    
    # Add samples
    num_added = dataset.add_samples(evaluation_results)
    logger.debug(f"Added {num_added} samples to training dataset")
    
    return num_added

def get_dataset_stats(
    dataset: TrainingDataset,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Get statistics about dataset
    
    Args:
        dataset: TrainingDataset object
        logger: Logger
    
    Returns:
        Dictionary with stats
    """
    if dataset.size() == 0:
        return {'size': 0, 'min_score': None, 'max_score': None, 'avg_score': None}
    
    scores = [s.get('actual_score', 0) for s in dataset.samples]
    
    stats = {
        'size': dataset.size(),
        'min_score': min(scores),
        'max_score': max(scores),
        'avg_score': sum(scores) / len(scores),
    }
    
    logger.debug(f"Dataset stats: {stats}")
    return stats

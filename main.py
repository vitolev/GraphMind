import logging
from pathlib import Path
from config.settings import Config
from pipeline.main_loop import run_pipeline

def setup_logging(config: Config) -> logging.Logger:
    """Set up logging to file and console"""
    config.log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("gnn_multiagent")
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(config.log_dir / f"{config.experiment_name}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def create_directories(config: Config) -> None:
    """Create required directories"""
    config.save_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)
    config.data_dir.mkdir(parents=True, exist_ok=True)

def main():
    # Load config
    config_path = Path("config/experiment_config.yaml")
    
    if not config_path.exists():
        print(f"Creating default config at {config_path}")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = Config()
        config.save_yaml(config_path)
    
    config = Config.from_yaml(config_path)
    
    # Set up logging
    logger = setup_logging(config)
    logger.info("=" * 60)
    logger.info("GNN-BASED MULTIAGENT OPTIMIZATION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Config loaded from: {config_path}")
    
    # Create directories
    create_directories(config)
    logger.info(f"Checkpoint dir: {config.save_dir}")
    logger.info(f"Data dir: {config.data_dir}")
    
    # Run pipeline
    logger.info("=" * 60)
    logger.info("STARTING PIPELINE")
    logger.info("=" * 60 + "\n")
    
    try:
        metrics_df = run_pipeline(config, logger)
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

        logger.info("\n" + "=" * 60)
        logger.info("RUNNING POST-PROCESSING ANALYSIS")
        logger.info("=" * 60 + "\n")
        
        from post_processing.analytics import run_analytics
        run_analytics(metrics_df, config, logger)
        
        logger.info("\n" + "=" * 60)
        logger.info("POST-PROCESSING COMPLETE")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

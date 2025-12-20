from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml

@dataclass
class Config:
    """Single configuration object for the entire pipeline"""
    
    # Graph Generation
    num_graphs_per_iteration: int = 10000
    generation_strategy: str = "random"
    max_nodes: int = 20
    max_depth: int = 2

    gnn_model_type: str = "linear_regression_gnn"
    data_format: str = "Data"  # Options: "Data" or "HeteroData"

    # GNN Configuration
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 3
    gnn_num_heads: int = 8
    gnn_dropout: float = 0.1
    gnn_learning_rate: float = 0.001
    gnn_batch_size: int = 32
    gnn_epochs: int = 50
    gnn_device: str = "cuda"
    
    # Selection Strategy
    top_k_to_keep: int = 100
    eval_k_best: int = 50
    
    # Evaluation
    num_eval_problems: int = 10
    eval_timeout_seconds: int = 30
    llm_model: str = "gpt-4"
    
    # LLM Provider Configuration
    llm_provider: str = "groq"  # Options: "groq", "local" (transformers), or "ollama"
    local_llm_model: str = "microsoft/Phi-3-mini-4k-instruct"  # HuggingFace model name for local inference
    local_llm_device: str = "auto"  # Options: "auto", "cuda", "mps", "cpu" - auto detects CUDA/MPS if available
    ollama_model: str = "llama3.2"  # Ollama model name (e.g., "llama3.2", "qwen2.5", "phi3", "tinyllama")
    ollama_base_url: str = "http://localhost:11434"  # Ollama API base URL
    
    # Pipeline Control
    max_iterations: int = 1000
    checkpoint_frequency: int = 5
    
    # Paths
    save_dir: Path = field(default_factory=lambda: Path("./checkpoints"))
    log_dir: Path = field(default_factory=lambda: Path("./logs"))
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    config_path: Path = field(default_factory=lambda: Path("./config/experiment_config.yaml"))
    
    # Experiment Info
    experiment_name: str = "gnn_multiagent_opt"
    seed: Optional[int] = None
    
    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load config from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        
        # Convert string paths to Path objects
        if 'save_dir' in data:
            data['save_dir'] = Path(data['save_dir'])
        if 'log_dir' in data:
            data['log_dir'] = Path(data['log_dir'])
        if 'data_dir' in data:
            data['data_dir'] = Path(data['data_dir'])
        
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        from dataclasses import asdict
        d = asdict(self)
        # Convert Path objects to strings for YAML
        for key in ['save_dir', 'log_dir', 'data_dir', 'config_path']:
            if key in d:
                d[key] = str(d[key])
        return d
    
    def save_yaml(self, path: Path) -> None:
        """Save config to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @property
    def analytics_dir(self) -> Path:
        """Directory for analytics outputs"""
        return self.log_dir / "analytics"

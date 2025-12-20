"""LLM provider setup and initialization (Groq, Ollama, Local)."""

import os
from typing import List, Dict, Optional, Tuple

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use environment variables directly

# Groq client pool for multiple API keys (loaded from .env file)
_groq_clients = []  # List of Groq client instances, one per API key
_current_client_index = 0  # Index of currently active client
_rate_limited_api_keys = set()  # Set of API key indices that are rate-limited
_api_key_last_model = {}  # Track last model used per API key (to avoid repeating when switching back)

# Local LLM state
_local_llm_model = None
_local_llm_tokenizer = None
_local_llm_device = None


def _get_local_llm_device():
    """Auto-detect best device for local LLM (prioritizes CUDA, then MPS, then CPU)"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon GPU acceleration
        return "cpu"
    except ImportError:
        return "cpu"


def _load_local_llm(model_name: str, device: str = "auto"):
    """Lazy load local LLM model and tokenizer"""
    global _local_llm_model, _local_llm_tokenizer, _local_llm_device
    
    if _local_llm_model is not None:
        return _local_llm_model, _local_llm_tokenizer, _local_llm_device
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        raise ImportError(
            "transformers and torch are required for local LLM. Install with: "
            "pip install transformers torch"
        )
    
    if device == "auto":
        device = _get_local_llm_device()
    
    _local_llm_device = device
    print(f"📦 Loading local LLM: {model_name} on {device}...")
    
    # Load tokenizer
    _local_llm_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if _local_llm_tokenizer.pad_token is None:
        _local_llm_tokenizer.pad_token = _local_llm_tokenizer.eos_token
    
    # Check if model name indicates quantization preference
    use_quantization = "-4bit" in model_name.lower() or "-8bit" in model_name.lower()
    
    # Load model with appropriate settings
    if device == "cuda":
        if use_quantization:
            # Try to use 4-bit quantization on CUDA if bitsandbytes is available
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                _local_llm_model = AutoModelForCausalLM.from_pretrained(
                    model_name.replace("-4bit", "").replace("-8bit", ""),
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            except ImportError:
                print("⚠️  bitsandbytes not available, loading full precision model")
                use_quantization = False
        
        if not use_quantization:
            _local_llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
    elif device == "mps":
        # Apple Silicon: Use float16 for better performance on MPS
        # Note: 4-bit quantization with bitsandbytes doesn't work well on MPS,
        # so we use smaller models or bfloat16 for speed
        dtype = torch.bfloat16 if hasattr(torch, 'bfloat16') and torch.backends.mps.is_available() else torch.float16
        _local_llm_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,  # bfloat16 or float16 for MPS (faster than float32)
            trust_remote_code=True
        ).to(device)
    else:
        # CPU: Use float32 (float16 not well supported on CPU)
        _local_llm_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).to(device)
    
    _local_llm_model.eval()  # Set to evaluation mode
    print(f"✓ Local LLM loaded successfully")
    
    return _local_llm_model, _local_llm_tokenizer, _local_llm_device


def initialize_groq_clients():
    """Initialize Groq client(s) from GROQ_API_KEYS in .env file (comma-separated)."""
    global _groq_clients, _current_client_index, _rate_limited_api_keys, _api_key_last_model
    
    from groq import Groq
    
    _groq_clients = []
    _current_client_index = 0
    _rate_limited_api_keys = set()
    _api_key_last_model = {}
    
    # Load comma-separated API keys from environment
    groq_api_keys_str = os.getenv("GROQ_API_KEYS", "").strip()
    if not groq_api_keys_str:
        print("⚠️  No Groq API keys found. Set GROQ_API_KEYS in .env file (comma-separated)")
        return
    
    api_keys = [key.strip() for key in groq_api_keys_str.split(",") if key.strip()]
    
    # Initialize clients
    for i, api_key in enumerate(api_keys):
        try:
            _groq_clients.append(Groq(api_key=api_key))
            _api_key_last_model[i] = None
            print(f"✓ Initialized Groq API key {i+1}/{len(api_keys)}")
        except Exception as e:
            print(f"⚠️  Failed to initialize Groq API key {i+1}: {e}")
    
    if len(_groq_clients) > 0:
        print(f"✓ Initialized {len(_groq_clients)} Groq API key(s)")


def get_next_available_client_and_model(model_pool: List[str], current_model: str = None):
    """Get the next available Groq client and model.
    
    Priority: Keep the same model for the agent type, only switch models if necessary.
    
    Args:
        model_pool: List of available models for this agent type
        current_model: Preferred model to use (priority is to keep this model)
    
    Returns:
        Tuple of (Groq client, model name), or (None, None) if no clients available
    """
    from evaluation.model_selection import is_model_rate_limited
    
    global _groq_clients, _current_client_index, _rate_limited_api_keys, _api_key_last_model
    
    if len(_groq_clients) == 0:
        return None, None
    
    # Filter out rate-limited API keys
    available_indices = [i for i in range(len(_groq_clients)) if i not in _rate_limited_api_keys]
    
    if len(available_indices) == 0:
        # All API keys are rate-limited, reset and use all
        print("⚠️  All Groq API keys are rate-limited. Resetting and trying all keys again...")
        _rate_limited_api_keys.clear()
        available_indices = list(range(len(_groq_clients)))
    
    # Priority 1: Try to use the same model (current_model) with a different API key
    if current_model and current_model in model_pool:
        # Find an API key that hasn't been rate-limited with this model
        for idx in available_indices:
            if not is_model_rate_limited(current_model, idx):
                # This API key is available and this model works on it
                _current_client_index = idx
                client = _groq_clients[_current_client_index]
                _api_key_last_model[_current_client_index] = current_model
                return client, current_model
        
        # If we get here, all API keys are rate-limited with this model
        # Fall through to try a different model
    
    # Priority 2: If preferred model not available, rotate to next API key and use first available model
    # Rotate to next available API key
    if _current_client_index not in available_indices:
        _current_client_index = available_indices[0]
    else:
        # Find next available API key after current
        current_pos = available_indices.index(_current_client_index)
        _current_client_index = available_indices[(current_pos + 1) % len(available_indices)]
    
    client = _groq_clients[_current_client_index]
    
    # Select model: prefer first model in pool, or next model if this API key used a different one before
    last_model = _api_key_last_model.get(_current_client_index)
    if last_model and last_model in model_pool and len(model_pool) > 1:
        # This API key used a different model before, try next one
        try:
            last_index = model_pool.index(last_model)
            model_index = (last_index + 1) % len(model_pool)
        except ValueError:
            model_index = 0
    else:
        # Use first model in pool (or preferred model if it's first)
        if current_model and current_model in model_pool:
            model_index = model_pool.index(current_model)
        else:
            model_index = 0
    
    selected_model = model_pool[model_index]
    _api_key_last_model[_current_client_index] = selected_model
    
    return client, selected_model


def mark_api_key_rate_limited(api_key_index: int):
    """Mark an API key as rate-limited."""
    global _rate_limited_api_keys
    _rate_limited_api_keys.add(api_key_index)
    print(f"⚠️  Groq API key {api_key_index + 1} marked as rate-limited, switching to next key...")


# Initialize Groq clients on import
initialize_groq_clients()


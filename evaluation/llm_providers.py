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
    
    Strategy:
    1. Always use the same model for an agent type (first model in pool)
    2. When switching API keys, keep the same model
    3. Only switch models when returning to an API key that was rate-limited with that model
    
    Args:
        model_pool: List of available models for this agent type (should have one model for now)
        current_model: Preferred model to use (should match agent type's model)
    
    Returns:
        Tuple of (Groq client, model name), or (None, None) if no clients available
    """
    from evaluation.model_selection import is_model_rate_limited
    
    global _groq_clients, _current_client_index, _rate_limited_api_keys, _api_key_last_model
    
    if len(_groq_clients) == 0:
        return None, None
    
    # Get the primary model for this agent type (first model in pool)
    primary_model = model_pool[0] if model_pool else None
    if not primary_model:
        raise ValueError("Model pool is empty")
    
    # Use primary model (or current_model if it matches)
    model_to_use = primary_model
    if current_model and current_model in model_pool:
        model_to_use = current_model
    
    # Filter out rate-limited API keys
    available_indices = [i for i in range(len(_groq_clients)) if i not in _rate_limited_api_keys]
    
    if len(available_indices) == 0:
        # All API keys are rate-limited, reset and use all
        print("⚠️  All Groq API keys are rate-limited. Resetting and trying all keys again...")
        _rate_limited_api_keys.clear()
        available_indices = list(range(len(_groq_clients)))
    
    # Priority 1: Try to use the primary model with an API key that hasn't been rate-limited
    # Prefer API keys that haven't been rate-limited with this model
    for idx in available_indices:
        if not is_model_rate_limited(model_to_use, idx):
            # This API key is available and this model works on it
            _current_client_index = idx
            client = _groq_clients[_current_client_index]
            _api_key_last_model[_current_client_index] = model_to_use
            return client, model_to_use
    
    # Priority 2: All preferred API keys are rate-limited with the primary model
    # Rotate to next API key (try ones that were rate-limited - they might have reset)
    # When we come back to an API key, we'll try the same model again
    # Only switch models if it fails again (handled in call_groq_with_retry)
    
    # Rotate to next API key
    all_indices = list(range(len(_groq_clients)))
    if _current_client_index in all_indices:
        current_pos = all_indices.index(_current_client_index)
        _current_client_index = all_indices[(current_pos + 1) % len(all_indices)]
    else:
        _current_client_index = all_indices[0]
    
    client = _groq_clients[_current_client_index]
    
    # Always use the primary model when coming back to an API key
    # The rate limit might have reset, so we try again with the same model
    # If it fails again, call_groq_with_retry will handle it and can switch models
    selected_model = model_to_use
    _api_key_last_model[_current_client_index] = selected_model
    
    return client, selected_model


def mark_api_key_rate_limited(api_key_index: int):
    """Mark an API key as rate-limited."""
    global _rate_limited_api_keys
    _rate_limited_api_keys.add(api_key_index)
    print(f"⚠️  Groq API key {api_key_index + 1} marked as rate-limited, switching to next key...")


# Initialize Groq clients on import
initialize_groq_clients()


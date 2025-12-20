"""Model pools and selection logic for LLM calls."""

from typing import List

# Track models that are currently rate-limited per API key (to avoid using same model+key combination)
_rate_limited_model_key_combos = set()  # Set of (api_key_index, model) tuples


def _get_groq_clients_count():
    """Get the number of Groq clients (to avoid circular import)."""
    try:
        # Import here to avoid circular dependency
        from evaluation.llm_providers import _groq_clients
        return len(_groq_clients)
    except ImportError:
        return 0


def mark_model_rate_limited(model: str, api_key_index: int = None):
    """Mark a model as rate-limited for a specific API key (or all keys if api_key_index is None)"""
    global _rate_limited_model_key_combos
    if api_key_index is not None:
        _rate_limited_model_key_combos.add((api_key_index, model))
    else:
        # Mark for all API keys
        num_clients = _get_groq_clients_count()
        for i in range(num_clients):
            _rate_limited_model_key_combos.add((i, model))


def is_model_rate_limited(model: str, api_key_index: int = None) -> bool:
    """Check if a model is rate-limited for a specific API key (or any key if api_key_index is None)"""
    if api_key_index is not None:
        return (api_key_index, model) in _rate_limited_model_key_combos
    else:
        # Check if rate-limited on any API key
        num_clients = _get_groq_clients_count()
        return any((i, model) in _rate_limited_model_key_combos for i in range(num_clients))


# Solver models - good for problem-solving tasks (prioritize 14.4K RPD models)
SOLVER_MODELS = [
    "llama-3.1-8b-instant",                    # Fast, good quality, 30 RPM, 14.4K RPD, 6K TPM
    "allam-2-7b",                              # Good quality, 30 RPM, 14.4K RPD, 6K TPM
]

# Extract topic models - good for analysis/understanding tasks (needs 800 tokens, so models must support >= 800)
# Only models with 14.4K RPD
EXTRACT_TOPIC_MODELS = [
    "llama-3.1-8b-instant",                    # Fast, good quality, 30 RPM, 14.4K RPD, supports 800+ tokens
    "allam-2-7b",                              # Good quality, 30 RPM, 14.4K RPD, supports 800+ tokens
]

# Validator models - good for validation/critical analysis (14.4K RPD models)
VALIDATOR_MODELS = [
    "llama-3.1-8b-instant",                    # Fast, good quality, 30 RPM, 14.4K RPD
    "allam-2-7b",                              # Good quality, 30 RPM, 14.4K RPD
]

# Combine all models - good for synthesis tasks (14.4K RPD models)
COMBINE_ALL_MODELS = [
    "llama-3.1-8b-instant",                    # Fast, good quality, 30 RPM, 14.4K RPD
    "allam-2-7b",                              # Good quality, 30 RPM, 14.4K RPD
]


def select_model_deterministic(problem_text: str, model_pool: List[str]) -> str:
    """
    Deterministically select a model from pool based on problem text.
    Uses first character of problem to ensure consistent selection.
    Avoids models that are rate-limited on ALL API keys (but will still try them if needed).
    
    Args:
        problem_text: The problem text to base selection on
        model_pool: List of model names to choose from
    
    Returns:
        Selected model name
    """
    if not problem_text:
        # Return first model that's not rate-limited on all API keys, or first model if all are
        for model in model_pool:
            if not is_model_rate_limited(model):  # Check if rate-limited on all keys
                return model
        return model_pool[0]
    
    # Filter out models that are rate-limited on ALL API keys
    available_models = [m for m in model_pool if not is_model_rate_limited(m)]
    if not available_models:
        # If all models are rate-limited everywhere, use the original pool anyway
        available_models = model_pool
    
    # Use first character (or first non-space char) to determine model
    first_char = problem_text.strip()[0].lower() if problem_text.strip() else 'a'
    
    # Convert character to index (0-25 for a-z, wraps around)
    char_index = ord(first_char) - ord('a')
    if char_index < 0 or char_index > 25:
        char_index = 0  # Default for non-letter characters
    
    # Select model based on character index from available models
    model_index = char_index % len(available_models)
    return available_models[model_index]


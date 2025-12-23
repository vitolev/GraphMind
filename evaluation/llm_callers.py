"""Functions for calling LLMs (Groq, Ollama, Local)."""

import os
import time
from typing import Dict, List, Optional

# Import requests for Ollama API (optional dependency)
try:
    import requests
except ImportError:
    requests = None  # Will fail gracefully if Ollama provider is used without requests

from evaluation.llm_providers import (
    _load_local_llm,
    get_next_available_client_and_model,
    mark_api_key_rate_limited,
)

# Import Groq clients state (needed for checking client count)
from evaluation import llm_providers
from evaluation.model_selection import (
    SOLVER_MODELS,
    EXTRACT_TOPIC_MODELS,
    VALIDATOR_MODELS,
    COMBINE_ALL_MODELS,
    mark_model_rate_limited,
    unmark_model_rate_limited,
    is_model_rate_limited,
)


# Global provider setting (can be set via environment variable LLM_PROVIDER or by calling set_llm_provider)
_llm_provider = os.getenv("LLM_PROVIDER", "groq").lower()  # Default to groq
_local_model_name = os.getenv("LOCAL_LLM_MODEL", "microsoft/Phi-3-mini-4k-instruct")
_local_device = os.getenv("LOCAL_LLM_DEVICE", "auto").lower()
_ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")


def set_llm_provider(provider: str, local_model: str = None, local_device: str = None, ollama_model: str = None, ollama_base_url: str = None):
    """Set the LLM provider globally. Call this before using LLM functions.
    
    Args:
        provider: "groq", "local", or "ollama"
        local_model: HuggingFace model name (only used if provider="local")
        local_device: "auto", "cuda", "mps", or "cpu" (only used if provider="local")
        ollama_model: Ollama model name (only used if provider="ollama")
        ollama_base_url: Ollama API base URL (only used if provider="ollama")
    """
    global _llm_provider, _local_model_name, _local_device, _ollama_model, _ollama_base_url
    _llm_provider = provider.lower()
    if local_model:
        _local_model_name = local_model
    if local_device:
        _local_device = local_device.lower()
    if ollama_model:
        _ollama_model = ollama_model
    if ollama_base_url:
        _ollama_base_url = ollama_base_url


def call_local_llm(messages: List[Dict], model_name: str, device: str = "auto", max_tokens: int = 128) -> str:
    """Call local LLM using transformers library"""
    model, tokenizer, device = _load_local_llm(model_name, device)
    
    # Convert messages to prompt format (chat template if available, else simple format)
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        # Use chat template if available (most modern models)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Fallback: simple format
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}\n")
            elif role == "user":
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        prompt = "".join(prompt_parts) + "\nAssistant:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate with optimized settings for speed
    import torch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            # Optimizations for speed
            use_cache=True,  # KV cache for faster generation
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def call_ollama(messages: List[Dict], model: str = None, max_tokens: int = 128, base_url: str = None) -> str:
    """Call Ollama API for local LLM inference
    
    Args:
        messages: List of message dicts with "role" and "content" keys
        model: Ollama model name (uses global setting if None)
        max_tokens: Maximum tokens to generate
        base_url: Ollama API base URL (uses global setting if None)
    
    Returns:
        Generated text response
    """
    global _ollama_model, _ollama_base_url
    
    if requests is None:
        raise ImportError(
            "requests library is required for Ollama provider. "
            "Install with: pip install requests"
        )
    
    if base_url is None:
        base_url = _ollama_base_url
    if model is None:
        model = _ollama_model
    
    # Convert messages format for Ollama API
    # Ollama expects a "messages" array with role and content
    ollama_messages = []
    for msg in messages:
        ollama_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    payload = {
        "model": model,
        "messages": ollama_messages,
        "stream": False,  # Disable streaming for cleaner JSON response
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
        }
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=300  # 5 minute timeout
        )
        response.raise_for_status()
        
        # Handle potential JSON parsing issues
        try:
            result = response.json()
        except ValueError as json_err:
            # If JSON parsing fails, try to extract text from response
            response_text = response.text
            raise ValueError(
                f"Failed to parse Ollama JSON response. "
                f"Response text (first 500 chars): {response_text[:500]}. "
                f"JSON error: {json_err}"
            )
        
        # Ollama returns {"message": {"role": "assistant", "content": "..."}}
        if "message" in result and "content" in result["message"]:
            return result["message"]["content"].strip()
        else:
            raise ValueError(f"Unexpected Ollama response format: {result}")
            
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Could not connect to Ollama at {base_url}. "
            "Make sure Ollama is installed and running. Install from: https://ollama.ai"
        )
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            raise RuntimeError(
                f"Model '{model}' not found in Ollama. "
                f"Install it with: ollama pull {model}"
            )
        raise


def call_groq_with_retry(messages: List[Dict], model: str = "llama-3.1-8b-instant", max_tokens: int = 128, max_retries: int = 5, model_pool: List[str] = None) -> str:
    """Generic Groq API call with retry logic and automatic API key + model rotation
    
    Args:
        messages: List of message dicts
        model: Model name (used to determine model pool if model_pool not provided)
        max_tokens: Maximum tokens to generate
        max_retries: Maximum retry attempts
        model_pool: Optional list of models to choose from. If None, determines pool based on model name.
    """
    if len(llm_providers._groq_clients) == 0:
        raise RuntimeError("Groq client not initialized. Set GROQ_API_KEYS in .env file (comma-separated).")
    
    # Determine model pool if not provided
    if model_pool is None:
        # Check which pool the model belongs to
        if model in SOLVER_MODELS:
            model_pool = SOLVER_MODELS
        elif model in EXTRACT_TOPIC_MODELS:
            model_pool = EXTRACT_TOPIC_MODELS
        elif model in VALIDATOR_MODELS:
            model_pool = VALIDATOR_MODELS
        elif model in COMBINE_ALL_MODELS:
            model_pool = COMBINE_ALL_MODELS
        else:
            # Default to SOLVER_MODELS
            model_pool = SOLVER_MODELS
    
    retry_delay = 2
    current_model = model
    
    for attempt in range(max_retries):
        # Get next available client and model (rotates API keys and models)
        client, selected_model = get_next_available_client_and_model(model_pool, current_model)
        
        if client is None:
            raise RuntimeError("No available Groq clients. All API keys may be rate-limited.")
        
        current_api_key_index = llm_providers._current_client_index
        
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=selected_model,
                max_completion_tokens=max_tokens,
            )
            # Success! If this API key was previously rate-limited with this model, unmark it
            # (the rate limit likely reset while we were using other API keys)
            unmark_model_rate_limited(selected_model, current_api_key_index)
            return chat_completion.choices[0].message.content
        except Exception as e:
            error_msg = str(e).lower()
            if 'rate limit' in error_msg or '429' in error_msg:
                # Check if we're coming back to an API key that was already rate-limited with this model
                was_already_rate_limited = is_model_rate_limited(selected_model, current_api_key_index)
                
                # Mark this specific API key + model combination as rate-limited
                mark_model_rate_limited(selected_model, current_api_key_index)
                
                # If this is the second failure on this API key (was already rate-limited),
                # and we have alternative models, switch to a different model
                if was_already_rate_limited and len(model_pool) > 1:
                    # This API key is still rate-limited after we came back to it
                    # Switch to a different model from the pool
                    try:
                        current_index = model_pool.index(selected_model)
                        next_index = (current_index + 1) % len(model_pool)
                        current_model = model_pool[next_index]
                        print(f"      🔄 API key {current_api_key_index + 1} still rate-limited with {selected_model} on second try, switching to {current_model}")
                    except ValueError:
                        pass  # Keep current_model as selected_model
                elif len(llm_providers._groq_clients) > 1:
                    # First time rate-limited on this API key, just switch API key (keep same model)
                    mark_api_key_rate_limited(current_api_key_index)
                    print(f"      ⏳ Rate limited on API key {current_api_key_index + 1} with model {selected_model}. Switching to next API key (keeping same model)...")
                else:
                    # Only one API key, wait and retry
                    mark_api_key_rate_limited(current_api_key_index)
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"      ⏳ Rate limited on {selected_model}. Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
            elif 'invalid_request_error' in error_msg or '400' in error_msg:
                # Check for organization_restricted error first
                if 'organization_restricted' in error_msg or 'Organization has been restricted' in error_msg:
                    api_key_num = current_api_key_index + 1  # Human-readable (1-indexed)
                    error_msg_text = (
                        f"ERROR: Organization restricted error on API key {api_key_num} (index {current_api_key_index})\n"
                        f"This means API key {api_key_num} in your GROQ_API_KEYS sequence has been restricted.\n"
                        f"Please remove API key {api_key_num} from your .env file's GROQ_API_KEYS\n"
                        f"If your keys are: key1,key2,key3,key4 and this is key {api_key_num}, remove the {api_key_num}th key\n"
                        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    )
                    
                    print(f"\n      ❌ {error_msg_text.split(chr(10))[0]}")
                    print(f"      🔑 {error_msg_text.split(chr(10))[1]}")
                    print(f"      💡 {error_msg_text.split(chr(10))[2]}")
                    print(f"      📝 {error_msg_text.split(chr(10))[3]}")
                    
                    # Save to file
                    from pathlib import Path
                    restricted_keys_file = Path("logs/restricted_api_keys.log")
                    restricted_keys_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(restricted_keys_file, "a") as f:
                        f.write("\n" + "="*60 + "\n")
                        f.write(error_msg_text)
                    
                    print(f"      💾 This error has been logged to: {restricted_keys_file}")
                    
                    # Mark this API key as restricted so we skip it in future calls
                    mark_api_key_rate_limited(current_api_key_index)
                    # Continue to next attempt (will try another API key)
                    continue
                
                # Other 400 errors - try combining messages
                if len(messages) > 1:
                    combined_text = ""
                    for msg in messages:
                        if msg["role"] == "system":
                            combined_text += msg["content"] + "\n\n"
                        elif msg["role"] == "user":
                            combined_text += msg["content"] + "\n\n"
                    
                    messages = [{"role": "user", "content": combined_text}]
                    
                    chat_completion = client.chat.completions.create(
                        messages=messages,
                        model=selected_model,
                        max_completion_tokens=max_tokens,
                    )
                    return chat_completion.choices[0].message.content
                else:
                    raise
            else:
                raise
    
    raise RuntimeError(f"Max retries ({max_retries}) reached")


def call_llm(messages: List[Dict], model: str = None, max_tokens: int = 128, max_retries: int = 5) -> str:
    """Unified LLM call function that routes to the appropriate provider.
    
    Args:
        messages: List of message dicts with "role" and "content" keys
        model: Model name (only used for Groq, ignored for local/Ollama)
        max_tokens: Maximum tokens to generate
        max_retries: Maximum retry attempts (only used for Groq)
    
    Returns:
        Generated text response
    """
    global _llm_provider, _local_model_name, _local_device
    
    if _llm_provider == "ollama":
        # Use Ollama - model parameter is ignored, use global setting
        return call_ollama(messages, None, max_tokens, None)
    elif _llm_provider == "local":
        # Use local transformers LLM - model parameter is ignored, use global setting
        return call_local_llm(messages, _local_model_name, _local_device, max_tokens)
    else:
        # Use Groq - model parameter is used
        if model is None:
            model = "llama-3.1-8b-instant"  # Default Groq model
        
        # Determine model pool based on model name
        if model in SOLVER_MODELS:
            model_pool = SOLVER_MODELS
        elif model in EXTRACT_TOPIC_MODELS:
            model_pool = EXTRACT_TOPIC_MODELS
        elif model in VALIDATOR_MODELS:
            model_pool = VALIDATOR_MODELS
        elif model in COMBINE_ALL_MODELS:
            model_pool = COMBINE_ALL_MODELS
        else:
            model_pool = SOLVER_MODELS  # Default
        
        return call_groq_with_retry(messages, model, max_tokens, max_retries, model_pool=model_pool)


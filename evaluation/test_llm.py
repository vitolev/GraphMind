import operator
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, TypedDict, Annotated
from groq import Groq
from langgraph.graph import StateGraph, START, END
import time

# Import requests for Ollama API (optional dependency)
try:
    import requests
except ImportError:
    requests = None  # Will fail gracefully if Ollama provider is used without requests

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use environment variables directly

# ============================================================================
# LLM SETUP - CONFIGURABLE PROVIDER (GROQ OR LOCAL)
# ============================================================================

# Groq client will automatically use GROQ_API_KEY environment variable (only if using Groq)
client = None
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

# Initialize Groq client only if we might use it
try:
    client = Groq()
except Exception:
    pass  # Will fail gracefully if GROQ_API_KEY not set

# ============================================================================
# MODEL POOLS FOR RATE LIMIT DISTRIBUTION
# ============================================================================

# Track models that are currently rate-limited (to avoid using them)
_rate_limited_models = set()

def mark_model_rate_limited(model: str):
    """Mark a model as rate-limited so we can avoid it temporarily"""
    _rate_limited_models.add(model)

def is_model_rate_limited(model: str) -> bool:
    """Check if a model is currently marked as rate-limited"""
    return model in _rate_limited_models

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
    Avoids models that are currently rate-limited.
    
    Args:
        problem_text: The problem text to base selection on
        model_pool: List of model names to choose from
    
    Returns:
        Selected model name
    """
    if not problem_text:
        # Return first non-rate-limited model, or first model if all are rate-limited
        for model in model_pool:
            if not is_model_rate_limited(model):
                return model
        return model_pool[0]
    
    # Filter out rate-limited models
    available_models = [m for m in model_pool if not is_model_rate_limited(m)]
    if not available_models:
        # If all models are rate-limited, use the original pool
        available_models = model_pool
    
    # Use first character (or first non-space char) to determine model
    first_char = problem_text.strip()[0].lower() if problem_text.strip() else 'a'
    
    # Convert character to index (0-25 for a-z, wraps around)
    char_index = ord(first_char) - ord('a')
    if char_index < 0 or char_index > 25:
        char_index = 0  # Default for non-letter characters
    
    # Select model based on character index from available (non-rate-limited) models
    model_index = char_index % len(available_models)
    return available_models[model_index]


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

def call_groq_with_retry(messages: List[Dict], model: str = "llama-3.1-8b-instant", max_tokens: int = 128, max_retries: int = 5) -> str:
    """Generic Groq API call with retry logic"""
    if client is None:
        raise RuntimeError("Groq client not initialized. Set GROQ_API_KEY environment variable or switch to local LLM.")
    
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model,
                max_completion_tokens=max_tokens,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            error_msg = str(e).lower()
            if 'rate limit' in error_msg or '429' in error_msg:
                # Mark this model as rate-limited
                mark_model_rate_limited(model)
                wait_time = retry_delay * (2 ** attempt)
                print(f"      ⏳ Rate limited on {model}. Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                print(f"      ⚠️  Model {model} marked as rate-limited, will avoid it for future requests")
                time.sleep(wait_time)
            elif 'invalid_request_error' in error_msg or '400' in error_msg:
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
                        model=model,
                        max_completion_tokens=max_tokens,
                    )
                    return chat_completion.choices[0].message.content
                else:
                    raise
            else:
                raise
    
    raise RuntimeError(f"Max retries ({max_retries}) reached")

def call_llm(messages: List[Dict], model: str = None, max_tokens: int = 128, max_retries: int = 5) -> str:
    """
    Unified LLM call function that routes to Groq, Ollama, or local transformers LLM based on provider setting.
    
    Args:
        messages: List of message dicts with "role" and "content" keys
        model: Model name (only used for Groq, ignored for Ollama and local transformers)
        max_tokens: Maximum tokens to generate
        max_retries: Maximum retry attempts (only for Groq)
    
    Returns:
        Generated text response
    """
    global _llm_provider, _local_model_name, _local_device, _ollama_model, _ollama_base_url
    
    if _llm_provider == "ollama":
        # Use Ollama - always use the configured Ollama model (ignore model parameter)
        # This is because Ollama uses different model names than Groq
        return call_ollama(messages, _ollama_model, max_tokens, _ollama_base_url)
    elif _llm_provider == "local":
        # Use local transformers LLM - model parameter is ignored, use global setting
        return call_local_llm(messages, _local_model_name, _local_device, max_tokens)
    else:
        # Use Groq - model parameter is used
        if model is None:
            model = "llama-3.1-8b-instant"  # Default Groq model
        return call_groq_with_retry(messages, model, max_tokens, max_retries)

# Backward compatibility: call_groq_with_retry is now an alias that routes through call_llm
# But we keep the original function for direct Groq calls when needed


# ============================================================================
# STATE DEFINITIONS
# ============================================================================

@dataclass
class GlobalKnowledge:
    entries: Dict[int, Any] = field(default_factory=dict)
    
    def add(self, node_id: int, data: Any):
        self.entries[node_id] = data
    
    def get(self, node_id: int) -> Optional[Any]:
        return self.entries.get(node_id)


class AgentState(TypedDict):
    problem: Annotated[list, operator.add]
    global_knowledge: GlobalKnowledge
    result: Annotated[list, operator.add]
    node_type: Optional[str]
    node_id: Optional[int]
    solution: Optional[str]


# ============================================================================
# NODE FUNCTIONS - DECENTRALIZED WITH OWN PROMPTS
# ============================================================================

def solver_node(state: AgentState) -> dict:
    """Solver node - generates concise solution (max 100 chars) and parses result"""
    node_id = state.get("node_id")
    problem_text = state['problem'][0] if isinstance(state['problem'], list) and state['problem'] else str(state['problem'])
    
    # Select model deterministically based on problem
    selected_model = select_model_deterministic(problem_text, SOLVER_MODELS)
    print(f"\n🔧 Solver-{node_id}: Quick solution (model: {selected_model})")
    
    # SOLVER SPECIFIC PROMPT
    system_prompt = """You are an expert problem solver. Your task is to:
        - Read the task carefully
        - Provide ONLY the final answer to the problem
        - The answer will be evaluated based on correctness
        - Keep your answer concise (maximum 100 characters)
        - Output MUST be wrapped in XML tags"""
    
    user_prompt = f"""Problem: {problem_text}
        Your answer MUST be in this exact format (max 100 characters inside tags):
        <SOLUTION>
        [Your answer here - just the final result]
        </SOLUTION>"""
    
    assistant_start = "<SOLUTION>"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_start}
    ]
    
    try:
        response = call_llm(messages, model=selected_model, max_tokens=150)
        final_response = assistant_start + response
        state['global_knowledge'].add(node_id, final_response)
        print(f"   ✓ Response: {final_response[:80]}...")
        
        # PARSE <SOLUTION> TAG
        import re
        solution_match = re.search(r'<SOLUTION>(.*?)</SOLUTION>', final_response, re.DOTALL)
        
        if solution_match:
            parsed_solution = solution_match.group(1).strip()
            print(f"   ✓ Parsed solution: {parsed_solution}")
            
            # UPDATE STATE WITH PARSED SOLUTION
            state['solution'] = parsed_solution
            
            return {"solution": parsed_solution}
        else:
            print(f"   ⚠️  No <SOLUTION> tag found in response")
            return {}
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        state['global_knowledge'].add(node_id, f"Error: {str(e)}")
        return {"result": [f"Error: {str(e)}"]}

def extract_topic_node(state: AgentState) -> dict:
    """Extract topic node - creates tree structure for hiring"""
    node_id = state.get("node_id")
    problem_text = state['problem'][0] if isinstance(state['problem'], list) and state['problem'] else str(state['problem'])
    
    # Select model deterministically based on problem
    selected_model = select_model_deterministic(problem_text, EXTRACT_TOPIC_MODELS)
    print(f"\n📖 Extract_topic-{node_id}: Building topic tree (model: {selected_model})")
    
    # EXTRACT_TOPIC SPECIFIC PROMPT
    system_prompt = """You are an expert at identifying key information and creating hierarchical structures.
        Your task is to:
        1. Read the input carefully
        2. Extract the main topic or central theme
        3. Identify 2-5 key subtopics and their relationships
        4. Create a TREE-INSPIRED structure showing how topics relate
        5. Format as a hiring guide - what roles/expertise would be needed to address each topic

        Be precise, hierarchical, and useful for team-building decisions."""
    
    user_prompt = f"""Content: {problem_text}

        Extract and structure the main topic in a tree format for hiring decisions:
        <TOPIC_TREE>
        MAIN_TOPIC: [Main topic here]
        ├─ SUBTOPIC_1: [First subtopic]
        │  └─ EXPERTISE_NEEDED: [What skills/roles needed]
        ├─ SUBTOPIC_2: [Second subtopic]
        │  └─ EXPERTISE_NEEDED: [What skills/roles needed]
        └─ SUBTOPIC_3: [Third subtopic]
        └─ EXPERTISE_NEEDED: [What skills/roles needed]
        HIRING_RECOMMENDATION: [What roles to hire]
        </TOPIC_TREE>"""
    
    assistant_start = "<TOPIC_TREE>\n"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_start}
    ]
    
    import time
    try:
        start_time = time.time()
        response = call_llm(messages, model=selected_model, max_tokens=800)
        elapsed_time = time.time() - start_time
        final_response = assistant_start + response
        state['global_knowledge'].add(node_id, final_response)
        print(f"   ✓ Response: {final_response[:80]}...")
        print(f"   ⏱️ Time taken: {elapsed_time:.2f} seconds")
        return {"result": [final_response]}
    except Exception as e:
        print(f"   ❌ Error: {e}")
        state['global_knowledge'].add(node_id, f"Error: {str(e)}")
        return {"result": [f"Error: {str(e)}"]}


def validator_node(state: AgentState) -> dict:
    """Validator node - critical analysis, looks for counter-examples"""
    node_id = state.get("node_id")
    
    # Get the most recent result from global knowledge to validate
    # Try to find results from previous nodes
    most_recent_result = None
    for check_id in range(node_id - 1, -1, -1):
        potential_result = state['global_knowledge'].get(check_id)
        if potential_result:
            most_recent_result = potential_result
            break
    
    if not most_recent_result:
        most_recent_result = state['problem'][0] if isinstance(state['problem'], list) and state['problem'] else str(state['problem'])
    
    # Select model deterministically based on original problem text
    problem_text = state['problem'][0] if isinstance(state['problem'], list) and state['problem'] else str(state['problem'])
    selected_model = select_model_deterministic(problem_text, VALIDATOR_MODELS)
    print(f"\n✓ Validator-{node_id}: Critical validation (model: {selected_model})")
    
    # VALIDATOR SPECIFIC PROMPT
    system_prompt = """You are a critical quality assurance expert. Your task is to:
        1. Review the provided solution or information
        2. Check for logical consistency and correctness
        3. Try to find counter-examples or edge cases that break the solution
        4. If you find ANY issue or counter-example, output FALSE
        5. If the solution is completely correct, output TRUE

        Be VERY critical and skeptical. Look for hidden errors."""
    
    user_prompt = f"""Solution to validate: {most_recent_result}

        Check this solution carefully. If it's correct in ALL cases, output:
        <VALIDATION>
        RESULT: TRUE
        REASONING: [Why this is correct]
        </VALIDATION>

        If you find ANY flaw or counter-example, output:
        <VALIDATION>
        RESULT: FALSE
        COUNTER_EXAMPLE: [Specific case where it fails]
        ISSUE: [Why this breaks the solution]
        </VALIDATION>"""
    
    assistant_start = "<VALIDATION>\n"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_start}
    ]
    
    try:
        response = call_llm(messages, model=selected_model, max_tokens=400)
        final_response = assistant_start + response
        state['global_knowledge'].add(node_id, final_response)
        print(f"   ✓ Response: {final_response[:80]}...")
        return {}
    except Exception as e:
        print(f"   ❌ Error: {e}")
        state['global_knowledge'].add(node_id, f"Error: {str(e)}")
        return {"result": [f"Error: {str(e)}"]}


def combine_all_node(state: AgentState, combine_all_edges: dict) -> dict:
    """Combine all node - synthesizes results from solver and validator"""
    node_id = state.get("node_id")
    
    # Get original problem text for deterministic model selection
    problem_text = state['problem'][0] if isinstance(state['problem'], list) and state['problem'] else str(state['problem'])
    selected_model = select_model_deterministic(problem_text, COMBINE_ALL_MODELS)
    print(f"\n🔗 Combine_all-{node_id}: Synthesizing all results (model: {selected_model})")
    
    incoming = combine_all_edges.get(node_id, [])
    
    # Collect results from incoming nodes
    collected_results = {}
    for inc_id in incoming:
        data = state['global_knowledge'].get(inc_id)
        if data:
            collected_results[inc_id] = str(data)[:500]
    
    results_text = "\n---\n".join([f"Node {nid}: {data}" for nid, data in collected_results.items()])
    
    # COMBINE_ALL SPECIFIC PROMPT
    system_prompt = """You are an expert synthesizer and decision-maker. Your task is to:
        1. Review all the provided results from different analysis nodes
        2. Compare different perspectives and solutions
        3. Synthesize findings into a FINAL VERDICT
        4. Create a clear recommendation
        5. Highlight any conflicts or important insights

        Be objective and create a clear final recommendation."""
    
    user_prompt = f"""Synthesis Request - Combine these results:

        {results_text}

        Create a final synthesis:
        <SYNTHESIS>
        FINAL_VERDICT: [Summary of findings]
        CONFIDENCE: [HIGH/MEDIUM/LOW]
        KEY_FINDINGS: [Summary of key points]
        RECOMMENDATION: [What to do based on results]
        </SYNTHESIS>"""
    
    assistant_start = "<SYNTHESIS>\n"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_start}
    ]
    
    try:
        response = call_llm(messages, model=selected_model, max_tokens=500)
        final_response = assistant_start + response
        state['global_knowledge'].add(node_id, final_response)
        print(f"   ✓ Response: {final_response[:80]}...")
        return {}
    except Exception as e:
        print(f"   ❌ Error: {e}")
        state['global_knowledge'].add(node_id, f"Error: {str(e)}")
        return {}

def split_node(state: AgentState) -> dict:
    """Split node - pass-through (does nothing)"""
    node_id = state.get("node_id")
    print(f"\n✂️  Split-{node_id}: Pass-through")
    return {}


def python_solver_node(state: AgentState) -> dict:
    """Python solver node - pass-through (does nothing)"""
    node_id = state.get("node_id")
    print(f"\n🐍 Python_solver-{node_id}: Pass-through")
    return {}


def decompose_node(state: AgentState) -> dict:
    """Decompose node - pass-through (does nothing)"""
    node_id = state.get("node_id")
    print(f"\n📋 Decompose-{node_id}: Pass-through")
    return {}


def explain_node(state: AgentState) -> dict:
    """Explain node - pass-through (does nothing)"""
    node_id = state.get("node_id")
    print(f"\n📖 Explain-{node_id}: Pass-through")
    return {}

# ============================================================================
# BUILD LANGGRAPH FROM STRUCTURE
# ============================================================================

def build_langgraph(nodes: List[Tuple[int, str]], edges: List[Tuple[int, int]]):
    """Build LangGraph using LangGraph's native routing"""
    
    id_to_type = {n: t for n, t in nodes}
    graph_out = {}
    graph_in = {}
    
    for src, dst in edges:
        if src not in graph_out:
            graph_out[src] = []
        if dst not in graph_in:
            graph_in[dst] = []
        graph_out[src].append(dst)
        graph_in[dst].append(src)
    
    combine_all_edges = {}
    for node_id, node_type in nodes:
        if node_type == "Combine_all":
            incoming = graph_in.get(node_id, [])
            combine_all_edges[node_id] = incoming
    
    print("🔨 Building LangGraph...")
    print(f"  Nodes: {len(nodes)}, Edges: {len(edges)}")
    
    builder = StateGraph(AgentState)
    
    node_handlers = {
        "Solver": solver_node,
        "Extract_topic": extract_topic_node,
        "Validator": validator_node,
        "Split": split_node,                  
        "Python_solver": python_solver_node,   
        "Decompose": decompose_node,          
        "Explain": explain_node,                

    }
    
    print("\n  Adding nodes:")
    
    for node_id, node_type in nodes:
        if node_type in ["START", "END"]:
            continue
        
        node_name = f"{node_type.lower()}_{node_id}"
        
        if node_type == "Combine_all":
            def make_combine_node(nid):
                def handler(state):
                    state["node_id"] = nid
                    return combine_all_node(state, combine_all_edges)
                return handler
            builder.add_node(node_name, make_combine_node(node_id))
        
        elif node_type in node_handlers:
            def make_typed_node(nid, ntype, handler_func):
                def handler(state):
                    state["node_id"] = nid
                    return handler_func(state)
                return handler
            builder.add_node(node_name, make_typed_node(node_id, node_type, node_handlers[node_type]))
        
        elif node_type in ["True_pass", "False_pass"]:
            def make_pass_node(nid, ntype):
                def handler(state):
                    print(f"\n{'✓' if ntype == 'True_pass' else '✗'} {ntype}-{nid}")
                    return {"result": []}
                return handler
            builder.add_node(node_name, make_pass_node(node_id, node_type))
        
        else:
            def make_generic_node(nid, ntype):
                def handler(state: AgentState) -> dict:
                    print(f"\n🔧 {ntype}-{nid}: Generic node")
                    problem_text = state['problem'][0] if isinstance(state['problem'], list) and state['problem'] else str(state['problem'])
                    return {"result": [f"Generic node {nid}: {problem_text[:50]}..."]}
                return handler
            builder.add_node(node_name, make_generic_node(node_id, node_type))
        
        print(f"    ✓ {node_name}")
    
    def get_node_name(node_id: int) -> Optional[str]:
        node_type = id_to_type.get(node_id)
        if node_type in ["START", "END"]:
            return None
        return f"{node_type.lower()}_{node_id}"
    
    print("\n  Adding edges:")
    
    start_node_id = None
    end_node_id = None
    for node_id, node_type in nodes:
        if node_type == "START":
            start_node_id = node_id
        elif node_type == "END":
            end_node_id = node_id
    
    if start_node_id is not None:
        for dst in graph_out.get(start_node_id, []):
            dst_name = get_node_name(dst)
            if dst_name:
                builder.add_edge(START, dst_name)
                print(f"    ✓ START → {dst_name}")
    
    for src, dst in edges:
        src_type = id_to_type.get(src)
        dst_type = id_to_type.get(dst)
        
        if src_type in ["START", "END"] or dst_type in ["START", "END"]:
            continue
        
        src_name = get_node_name(src)
        dst_name = get_node_name(dst)
        
        if src_name and dst_name:
            builder.add_edge(src_name, dst_name)
            print(f"    ✓ {src_name} → {dst_name}")
    
    if end_node_id is not None:
        for src in graph_in.get(end_node_id, []):
            src_name = get_node_name(src)
            if src_name:
                builder.add_edge(src_name, END)
                print(f"    ✓ {src_name} → END")
    
    print("\n✓ LangGraph compilation complete\n")
    return builder.compile()


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_graph_ascii(graph):
    """Print ASCII representation of the graph"""
    print("\n" + "="*80)
    print("GRAPH STRUCTURE (ASCII)")
    print("="*80)
    try:
        print(graph.get_graph().draw_ascii())
    except Exception as e:
        print(f"Could not generate ASCII visualization: {e}")


# ============================================================================
# EXAMPLES
# ============================================================================

EXAMPLE_1 = {
    "name": "Linear Pipeline",
    "nodes": [
        (0, "START"),
        (1, "Extract_topic"),
        (2, "Solver"),
        (3, "Validator"),
        (4, "END")
    ],
    "edges": [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4)
    ]
}

EXAMPLE_2 = {
    "name": "With Combine (Parallel Execution)",
    "nodes": [
        (0, "START"),
        (1, "Extract_topic"),
        (2, "Solver"),
        (3, "Validator"),
        (4, "Combine_all"),
        (5, "END")
    ],
    "edges": [
        (0, 1),
        (0, 2),
        (2, 3),
        (1, 4),
        (3, 4),
        (4, 5)
    ]
}

# ============================================================================
# MAIN
# ============================================================================

def run_example(example: Dict[str, Any], problem: str):
    """Run example"""
    
    print("\n" + "="*80)
    print(f"EXAMPLE: {example['name']}")
    print("="*80)
    
    graph = build_langgraph(example["nodes"], example["edges"])
    
    visualize_graph_ascii(graph)
    
    initial_state: AgentState = {
        "problem": [problem],
        "global_knowledge": GlobalKnowledge(),
        "result": [],
        "node_type": None,
        "node_id": None,
        "solution": None
    }
    
    print("\nEXECUTING LANGGRAPH\n")
    
    try:
        result = graph.invoke(initial_state)
        
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        
        results = result['result'] if isinstance(result['result'], list) else [result['result']]
        print(f"\nFinal results ({len(results)} items):")
        for i, res in enumerate(results):
            print(f"\n  [{i}]:\n{str(res)}")
        
        print(f"\n\nAll execution entries:")
        for node_id, data in result['global_knowledge'].entries.items():
            print(f"\n  Node {node_id}:\n{str(data)[:300]}")
        
        return result
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run examples"""
    
    print("\n" + "="*80)
    print("LANGGRAPH WITH GROQ - DECENTRALIZED LLM CALLS")
    print("="*80)
    print("\nFeatures:")
    print("  ✓ Each node has its own prompt generation")
    print("  ✓ Decentralized LLM calls")
    print("  ✓ Retry logic with exponential backoff")
    print("  ✓ Solver: Fast, concise answers (max 100 chars)")
    print("  ✓ Extract_topic: Tree-based hiring guide")
    print("  ✓ Validator: Critical, finds counter-examples")
    print("  ✓ Combine_all: Synthesizes all results")
    print("="*80)
    
    run_example(EXAMPLE_1, "Solve: 2x + 5 = 13")
    
    print("\n" + "="*80)
    print("✓ EXAMPLE COMPLETED")
    print("="*80)


if __name__ == "__main__":
    #main()
    from graph_generation.graph_generation import _random_graph
    import random
    random.seed(49)
    number_of_nodes = []
    for i in range(100):
        print(f"\n=== Random Graph {i+1} ===")
        print(number_of_nodes)
        g = _random_graph(max_depth=3)
        if number_of_nodes.count(len(g.get_nodes())) <= 5 and len(g.get_nodes()) < 10 and len(number_of_nodes) < 30:
            print(g.get_nodes(), g.get_edges())
            results = run_example({"name": f"Example{i}", "nodes": g.get_nodes(), "edges": g.get_edges()}, "What is machine the largest commond divisor of 108984222 and 29245674?")
            print(f"Results: {results}")
            #g.visualize()
            number_of_nodes.append(len(g.get_nodes()))
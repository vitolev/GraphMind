"""LangGraph node functions for multi-agent system."""

import re
import time
from typing import Dict

from evaluation.agent_state import AgentState
from evaluation.llm_callers import call_llm
from evaluation.model_selection import (
    SOLVER_MODELS,
    EXTRACT_TOPIC_MODELS,
    VALIDATOR_MODELS,
    COMBINE_ALL_MODELS,
    select_model_deterministic,
)


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


def combine_all_node(state: AgentState, combine_all_edges: Dict) -> dict:
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


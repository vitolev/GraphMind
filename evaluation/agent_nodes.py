"""LangGraph node functions for multi-agent system."""

import re
import time
import subprocess
import sys
from typing import Dict, Optional, List, Any

from evaluation.agent_state import AgentState
from evaluation.llm_callers import call_llm
from evaluation.model_selection import (
    SOLVER_MODELS,
    EXTRACT_TOPIC_MODELS,
    VALIDATOR_MODELS,
    COMBINE_ALL_MODELS,
    select_model_deterministic,
)


def _get_node_scope(state: AgentState, node_id: int) -> str:
    """Determine which scope a node belongs to."""
    # Check if scope is explicitly mapped (e.g., by split node)
    scope_mapping = state.get("scope_mapping", {})
    if node_id in scope_mapping:
        return scope_mapping[node_id]
    
    # Otherwise, inherit from incoming edges
    graph = state.get("graph_structure")
    if graph:
        incoming = graph.get_incoming_nodes(node_id)
        if incoming:
            # Get scope from first incoming node
            # Find which scope contains knowledge from that node
            for scope_id, scope_knowledge in state.get("scoped_knowledge", {}).items():
                if incoming[0] in scope_knowledge.node_data:
                    return scope_id
    
    # Default to current scope
    return state.get("current_scope", "root")


def _get_incoming_knowledge(state: AgentState, node_id: int, scope_id: str) -> Dict[int, Any]:
    """Get knowledge from incoming nodes within the specified scope."""
    graph = state.get("graph_structure")
    if not graph:
        return {}
    
    incoming_nodes = graph.get_incoming_nodes(node_id)
    scoped_knowledge = state.get("scoped_knowledge", {})
    
    # First try the node's own scope
    scope_knowledge = scoped_knowledge.get(scope_id)
    
    relevant_knowledge = {}
    if scope_knowledge:
        for inc_id in incoming_nodes:
            data = scope_knowledge.get(inc_id)
            if data is not None:
                relevant_knowledge[inc_id] = data
    
    # If scope is empty, also check parent scope (for nodes after split)
    # For example, if scope is "root_split_1", also check "root"
    if not relevant_knowledge and "_split_" in scope_id:
        parent_scope = scope_id.rsplit("_split_", 1)[0]
        parent_knowledge = scoped_knowledge.get(parent_scope)
        if parent_knowledge:
            for inc_id in incoming_nodes:
                data = parent_knowledge.get(inc_id)
                if data is not None:
                    relevant_knowledge[inc_id] = data
    
    return relevant_knowledge


def _store_knowledge(state: AgentState, node_id: int, scope_id: str, data: Any):
    """Store knowledge from a node in its scope."""
    scoped_knowledge = state.get("scoped_knowledge", {})
    if scope_id not in scoped_knowledge:
        from evaluation.agent_state import ScopedKnowledge
        scoped_knowledge[scope_id] = ScopedKnowledge(scope_id=scope_id)
        state["scoped_knowledge"] = scoped_knowledge
    
    scoped_knowledge[scope_id].add(node_id, data)
    # Also update legacy global_knowledge for backward compatibility
    if "global_knowledge" in state:
        state["global_knowledge"].add(node_id, data)


def solver_node(state: AgentState) -> dict:
    """Solver node - generates solution and saves in structured, parseable format.
    
    Always saves output in <SOLVER_OUTPUT> tags so Combine_all can easily collect
    and merge all solver opinions. This ensures GNN learns that solvers are final outputs.
    """
    node_id = state.get("node_id")
    
    # Determine scope for this node
    scope_id = _get_node_scope(state, node_id)
    
    # Get incoming knowledge (from nodes in same scope)
    incoming_knowledge = _get_incoming_knowledge(state, node_id, scope_id)
    
    # Get problem text (from incoming nodes in scope, or original problem)
    problem_text = state['problem'][0] if isinstance(state['problem'], list) and state['problem'] else str(state['problem'])
    graph = state.get("graph_structure")
    
    # Check for Python_solver output to optionally include
    python_output = None
    if incoming_knowledge:
        for inc_id, inc_data in incoming_knowledge.items():
            inc_type = graph.get_node_type(inc_id) if graph else None
            
            # Check if Python_solver output is available
            if inc_type == "Python_solver":
                # Try to extract Python output
                python_match = re.search(r'<PYTHON_OUTPUT>(.*?)</PYTHON_OUTPUT>', str(inc_data), re.DOTALL)
                if python_match:
                    python_output = python_match.group(1).strip()
                    print(f"   💻 Using Python_solver output: {python_output[:50]}...")
            
            # If coming from decompose or split, use that as the problem
            if inc_type in ["Decompose", "Split"]:
                problem_text = str(inc_data)
                break
            else:
                # Otherwise, use as context but keep original problem
                problem_text = str(inc_data) if not problem_text else problem_text
                break
    
    # Select model deterministically based on problem
    selected_model = select_model_deterministic(problem_text, SOLVER_MODELS)
    print(f"\n🔧 Solver-{node_id}: Generating solution (model: {selected_model})")
    print(f"   📍 Scope: {scope_id}")
    if incoming_knowledge:
        print(f"   📥 Knowledge from: {list(incoming_knowledge.keys())}")
    
    # SOLVER SPECIFIC PROMPT
    system_prompt = """You are an expert problem solver. Your task is to:
        - Read the task carefully
        - Provide ONLY the final answer to the problem
        - The answer will be evaluated based on correctness
        - Keep your answer concise (maximum 100 characters)
        - Output MUST be wrapped in XML tags"""
    
    # Include Python output if available
    context_text = ""
    if python_output:
        context_text = f"\n\nPython execution result: {python_output}\nUse this information to help solve the problem."
    
    user_prompt = f"""Problem: {problem_text}{context_text}

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
        
        # PARSE <SOLUTION> TAG
        solution_match = re.search(r'<SOLUTION>(.*?)</SOLUTION>', final_response, re.DOTALL)
        
        if solution_match:
            parsed_solution = solution_match.group(1).strip()
            print(f"   ✓ Parsed solution: {parsed_solution}")
            
            # ALWAYS save in structured format for Combine_all to collect
            # Format: <SOLVER_OUTPUT>solution</SOLVER_OUTPUT>
            structured_output = f"<SOLVER_OUTPUT>{parsed_solution}</SOLVER_OUTPUT>"
            
            # Store in scoped knowledge (this is what Combine_all will collect)
            _store_knowledge(state, node_id, scope_id, structured_output)
            
            # Only update global solution if we're in root scope (single solver path)
            # Otherwise, Combine_all will set the final solution
            if scope_id == "root" or not scope_id.startswith("root_"):
                state['solution'] = parsed_solution
                return {"solution": parsed_solution}
            else:
                # In decomposed/split scope - save structured output, let Combine_all merge
                return {}
        else:
            print(f"   ⚠️  No <SOLUTION> tag found in response, storing raw response")
            # Still store raw response for debugging
            structured_output = f"<SOLVER_OUTPUT>{final_response}</SOLVER_OUTPUT>"
            _store_knowledge(state, node_id, scope_id, structured_output)
            return {}
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        error_output = f"<SOLVER_OUTPUT>ERROR: {str(e)}</SOLVER_OUTPUT>"
        _store_knowledge(state, node_id, scope_id, error_output)
        return {}


def extract_topic_node(state: AgentState) -> dict:
    """Extract topic node - creates tree structure for hiring"""
    node_id = state.get("node_id")
    
    # Determine scope for this node
    scope_id = _get_node_scope(state, node_id)
    
    # Get incoming knowledge (from nodes in same scope)
    incoming_knowledge = _get_incoming_knowledge(state, node_id, scope_id)
    
    # Get problem text (from incoming nodes in scope, or original problem)
    problem_text = state['problem'][0] if isinstance(state['problem'], list) and state['problem'] else str(state['problem'])
    graph = state.get("graph_structure")
    if incoming_knowledge:
        # Use knowledge from incoming nodes if available (this could be a subproblem from decompose)
        for inc_id, inc_data in incoming_knowledge.items():
            inc_type = graph.get_node_type(inc_id) if graph else None
            # If coming from decompose or split, use that as the problem
            if inc_type in ["Decompose", "Split"]:
                problem_text = str(inc_data)
                break
            else:
                # Otherwise, use as context but keep original problem
                problem_text = str(inc_data) if not problem_text else problem_text
                break
    
    # Select model deterministically based on problem
    selected_model = select_model_deterministic(problem_text, EXTRACT_TOPIC_MODELS)
    print(f"\n📖 Extract_topic-{node_id}: Building topic tree (model: {selected_model})")
    print(f"   📍 Scope: {scope_id}")
    if incoming_knowledge:
        print(f"   📥 Knowledge from: {list(incoming_knowledge.keys())}")
    
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
        
        # Parse topic tree
        topic_tree_match = re.search(r'<TOPIC_TREE>(.*?)</TOPIC_TREE>', final_response, re.DOTALL)
        
        if topic_tree_match:
            topic_tree_content = topic_tree_match.group(1).strip()
            
            # Extract MAIN_TOPIC
            main_topic_match = re.search(r'MAIN_TOPIC:\s*(.+?)(?=\n├─|\n└─|HIRING_RECOMMENDATION|$)', topic_tree_content, re.DOTALL)
            main_topic = main_topic_match.group(1).strip() if main_topic_match else "Not found"
            
            # Extract HIRING_RECOMMENDATION
            hiring_match = re.search(r'HIRING_RECOMMENDATION:\s*(.+?)$', topic_tree_content, re.DOTALL)
            hiring_rec = hiring_match.group(1).strip() if hiring_match else "Not specified"
            
            # Extract subtopics (simplified - just get first few lines)
            subtopics = []
            for line in topic_tree_content.split('\n'):
                if '├─' in line or '└─' in line:
                    subtopic_clean = re.sub(r'^.*?(├─|└─)\s*', '', line).strip()
                    if subtopic_clean and not subtopic_clean.startswith('EXPERTISE_NEEDED'):
                        subtopics.append(subtopic_clean[:100])  # Limit length
                        if len(subtopics) >= 5:  # Limit number
                            break
            
            parsed_result = f"MAIN_TOPIC: {main_topic}\nSUBTOPICS: {', '.join(subtopics[:3]) if subtopics else 'None'}\nHIRING_RECOMMENDATION: {hiring_rec[:200]}"
            print(f"   ✓ Parsed topic tree")
            print(f"   📌 Main topic: {main_topic[:60]}...")
            print(f"   📋 Subtopics: {len(subtopics)} found")
            
            # Store parsed topic tree in structured format
            structured_output = f"<TOPIC_TREE_OUTPUT>{parsed_result}</TOPIC_TREE_OUTPUT>"
            _store_knowledge(state, node_id, scope_id, structured_output)
        else:
            print(f"   ⚠️  No <TOPIC_TREE> tag found, storing raw response")
            structured_output = f"<TOPIC_TREE_OUTPUT>{final_response}</TOPIC_TREE_OUTPUT>"
            _store_knowledge(state, node_id, scope_id, structured_output)
        
        print(f"   ⏱️ Time taken: {elapsed_time:.2f} seconds")
        return {}
    except Exception as e:
        print(f"   ❌ Error: {e}")
        error_output = f"<TOPIC_TREE_OUTPUT>ERROR: {str(e)}</TOPIC_TREE_OUTPUT>"
        _store_knowledge(state, node_id, scope_id, error_output)
        return {}


def validator_node(state: AgentState) -> dict:
    """Validator node - critical analysis, looks for counter-examples"""
    node_id = state.get("node_id")
    
    # Determine scope for this node
    scope_id = _get_node_scope(state, node_id)
    
    # Get incoming knowledge (from nodes in same scope)
    incoming_knowledge = _get_incoming_knowledge(state, node_id, scope_id)
    
    # Get the most recent result from incoming nodes to validate
    most_recent_result = None
    if incoming_knowledge:
        # Use knowledge from the most recent incoming node
        for inc_id, inc_data in incoming_knowledge.items():
            most_recent_result = str(inc_data)
    
    if not most_recent_result:
        most_recent_result = state['problem'][0] if isinstance(state['problem'], list) and state['problem'] else str(state['problem'])
    
    # Select model deterministically based on original problem text
    problem_text = state['problem'][0] if isinstance(state['problem'], list) and state['problem'] else str(state['problem'])
    selected_model = select_model_deterministic(problem_text, VALIDATOR_MODELS)
    print(f"\n✓ Validator-{node_id}: Critical validation (model: {selected_model})")
    print(f"   📍 Scope: {scope_id}")
    if incoming_knowledge:
        print(f"   📥 Knowledge from: {list(incoming_knowledge.keys())}")
    
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
        
        # Parse validation result
        validation_match = re.search(r'<VALIDATION>(.*?)</VALIDATION>', final_response, re.DOTALL)
        
        if validation_match:
            validation_content = validation_match.group(1).strip()
            
            # Extract RESULT (TRUE/FALSE)
            result_match = re.search(r'RESULT:\s*(TRUE|FALSE)', validation_content, re.IGNORECASE)
            is_valid = result_match.group(1).upper() == "TRUE" if result_match else None
            
            # Extract REASONING or COUNTER_EXAMPLE
            reasoning_match = re.search(r'(?:REASONING|COUNTER_EXAMPLE|ISSUE):\s*(.+?)(?=(?:REASONING|COUNTER_EXAMPLE|ISSUE|$))', validation_content, re.DOTALL | re.IGNORECASE)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
            
            parsed_result = f"VALIDATION_RESULT: {is_valid}\nREASONING: {reasoning[:200]}"
            print(f"   ✓ Parsed validation: {is_valid}")
            print(f"   📝 Reasoning: {reasoning[:80]}...")
            
            # Store parsed validation in structured format
            structured_output = f"<VALIDATION_OUTPUT>{parsed_result}</VALIDATION_OUTPUT>"
            _store_knowledge(state, node_id, scope_id, structured_output)
        else:
            print(f"   ⚠️  No <VALIDATION> tag found, storing raw response")
            structured_output = f"<VALIDATION_OUTPUT>{final_response}</VALIDATION_OUTPUT>"
            _store_knowledge(state, node_id, scope_id, structured_output)
        
        return {}
    except Exception as e:
        print(f"   ❌ Error: {e}")
        error_output = f"<VALIDATION_OUTPUT>ERROR: {str(e)}</VALIDATION_OUTPUT>"
        _store_knowledge(state, node_id, scope_id, error_output)
        return {}


def combine_all_node(state: AgentState, combine_all_edges: Dict) -> dict:
    """Combine all node - collects all solver outputs and synthesizes final solution.
    
    This node:
    1. Collects all SOLVER_OUTPUT from incoming nodes across all scopes
    2. Extracts the actual solutions from structured format
    3. Synthesizes them into a single final solution
    4. Saves the final solution in parseable format
    """
    node_id = state.get("node_id")
    
    # Combine_all typically merges results back to root scope
    scope_id = "root"
    
    incoming = combine_all_edges.get(node_id, [])
    
    # Collect results from incoming nodes across ALL scopes
    collected_results = {}
    solver_solutions = []  # Extract all solver solutions for merging
    scoped_knowledge = state.get("scoped_knowledge", {})
    graph = state.get("graph_structure")
    
    for inc_id in incoming:
        # Search across all scopes for this node's output
        found = False
        for scope, scope_knowledge in scoped_knowledge.items():
            data = scope_knowledge.get(inc_id)
            if data:
                data_str = str(data)
                collected_results[inc_id] = {
                    "data": data_str[:500],
                    "scope": scope
                }
                
                # Extract solver solutions from structured format
                # Look for <SOLVER_OUTPUT> tags
                solver_match = re.search(r'<SOLVER_OUTPUT>(.*?)</SOLVER_OUTPUT>', data_str, re.DOTALL)
                if solver_match:
                    solution = solver_match.group(1).strip()
                    inc_type = graph.get_node_type(inc_id) if graph else "unknown"
                    solver_solutions.append({
                        "node_id": inc_id,
                        "node_type": inc_type,
                        "solution": solution,
                        "scope": scope
                    })
                    print(f"   ✓ Found solver solution from {inc_type}-{inc_id}: {solution[:50]}...")
                
                found = True
                break
        
        if not found:
            # Fallback to global_knowledge (legacy)
            data = state.get("global_knowledge", {}).get(inc_id)
            if data:
                collected_results[inc_id] = {
                    "data": str(data)[:500],
                    "scope": "unknown"
                }
    
    # Get original problem text for deterministic model selection
    problem_text = state['problem'][0] if isinstance(state['problem'], list) and state['problem'] else str(state['problem'])
    selected_model = select_model_deterministic(problem_text, COMBINE_ALL_MODELS)
    print(f"\n🔗 Combine_all-{node_id}: Merging {len(solver_solutions)} solver solution(s) (model: {selected_model})")
    print(f"   📍 Scope: {scope_id} (merged)")
    if collected_results:
        print(f"   📥 Combining from: {[(nid, info['scope']) for nid, info in collected_results.items()]}")
    
    # Build synthesis prompt
    if solver_solutions:
        # Focus on merging solver solutions
        solutions_text = "\n---\n".join([
            f"{sol['node_type']}-{sol['node_id']} (from scope '{sol['scope']}'): {sol['solution']}"
            for sol in solver_solutions
        ])
        
        system_prompt = """You are an expert synthesizer. Your task is to:
            1. Review all the solver solutions provided
            2. If they agree, confirm the common answer
            3. If they differ, choose the most correct one or synthesize them
            4. Output ONLY the final solution, nothing else
            
            Output MUST be in this exact format:
            <FINAL_SOLUTION>
            [Your final synthesized solution here]
            </FINAL_SOLUTION>"""
        
        user_prompt = f"""Problem: {problem_text}

            Here are {len(solver_solutions)} different solver solutions:
            
            {solutions_text}
            
            Synthesize these into ONE final solution. If solutions agree, use that answer.
            If they differ, choose the best one or combine them appropriately.
            
            Output ONLY the final solution in this format:
            <FINAL_SOLUTION>
            [Final solution here]
            </FINAL_SOLUTION>"""
    else:
        # Fallback: synthesize all collected results
        results_text = "\n---\n".join([
            f"Node {nid} (scope: {info['scope']}): {info['data']}" 
            for nid, info in collected_results.items()
        ])
        
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
    
    assistant_start = "<FINAL_SOLUTION>" if solver_solutions else "<SYNTHESIS>\n"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_start}
    ]
    
    try:
        response = call_llm(messages, model=selected_model, max_tokens=500)
        final_response = assistant_start + response
        
        # Parse final solution
        if solver_solutions:
            solution_match = re.search(r'<FINAL_SOLUTION>(.*?)</FINAL_SOLUTION>', final_response, re.DOTALL)
            if solution_match:
                final_solution = solution_match.group(1).strip()
                print(f"   ✓ Final synthesized solution: {final_solution}")
                
                # Store in root scope and update global solution
                structured_output = f"<SOLVER_OUTPUT>{final_solution}</SOLVER_OUTPUT>"
                _store_knowledge(state, node_id, scope_id, structured_output)
                
                # Update global solution
                state['solution'] = final_solution
                return {"solution": final_solution}
            else:
                print(f"   ⚠️  Could not parse final solution, storing raw response")
                _store_knowledge(state, node_id, scope_id, final_response)
                return {}
        else:
            # Store synthesis
            _store_knowledge(state, node_id, scope_id, final_response)
            print(f"   ✓ Response: {final_response[:80]}...")
            return {}
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        _store_knowledge(state, node_id, scope_id, f"Error: {str(e)}")
        return {}


def split_node(state: AgentState) -> dict:
    """Split node - creates new scopes for each outgoing branch"""
    node_id = state.get("node_id")
    scope_id = _get_node_scope(state, node_id)
    
    graph = state.get("graph_structure")
    if not graph:
        print(f"\n✂️  Split-{node_id}: No graph structure available")
        return {}
    
    # Get outgoing edges from this split node
    outgoing_nodes = graph.get_outgoing_nodes(node_id)
    
    # Create new scopes for each outgoing branch
    scope_mapping = state.get("scope_mapping", {})
    scoped_knowledge = state.get("scoped_knowledge", {})
    
    new_scopes = {}
    for i, target_node_id in enumerate(outgoing_nodes):
        new_scope_id = f"{scope_id}_split_{i+1}"
        new_scopes[target_node_id] = new_scope_id
        scope_mapping[target_node_id] = new_scope_id
        
        # Initialize empty knowledge for new scope
        if new_scope_id not in scoped_knowledge:
            from evaluation.agent_state import ScopedKnowledge
            scoped_knowledge[new_scope_id] = ScopedKnowledge(scope_id=new_scope_id)
    
    state["scope_mapping"] = scope_mapping
    state["scoped_knowledge"] = scoped_knowledge
    
    print(f"\n✂️  Split-{node_id}: Creating {len(new_scopes)} new scopes")
    print(f"   📍 Scope: {scope_id}")
    print(f"   🆕 New scopes: {new_scopes}")
    
    # Store split information in current scope
    _store_knowledge(state, node_id, scope_id, f"Split into {len(new_scopes)} branches: {list(new_scopes.values())}")
    
    return {}


def python_solver_node(state: AgentState) -> dict:
    """Python solver node - generates Python code, executes it, and saves output.
    
    This node:
    1. Gets the problem from incoming knowledge or state
    2. Generates Python code to solve the problem
    3. Executes the code with timeout and captures output
    4. Saves result in <PYTHON_OUTPUT> tags for Solver to optionally use
    """
    node_id = state.get("node_id")
    scope_id = _get_node_scope(state, node_id)
    incoming_knowledge = _get_incoming_knowledge(state, node_id, scope_id)
    
    # Get problem text (from incoming nodes in scope, or original problem)
    problem_text = state['problem'][0] if isinstance(state['problem'], list) and state['problem'] else str(state['problem'])
    graph = state.get("graph_structure")
    
    if incoming_knowledge:
        for inc_id, inc_data in incoming_knowledge.items():
            inc_type = graph.get_node_type(inc_id) if graph else None
            # If coming from decompose or split, use that as the problem
            if inc_type in ["Decompose", "Split"]:
                problem_text = str(inc_data)
                break
            else:
                # Otherwise, use as context but keep original problem
                problem_text = str(inc_data) if not problem_text else problem_text
                break
    
    # Select model deterministically based on problem
    selected_model = select_model_deterministic(problem_text, SOLVER_MODELS)  # Use solver models for code generation
    print(f"\n🐍 Python_solver-{node_id}: Generating and executing Python code (model: {selected_model})")
    print(f"   📍 Scope: {scope_id}")
    if incoming_knowledge:
        print(f"   📥 Knowledge from: {list(incoming_knowledge.keys())}")
    
    # PYTHON_SOLVER SPECIFIC PROMPT
    system_prompt = """You are an expert Python programmer. Your task is to:
        - Read the problem carefully
        - Generate Python code that solves the problem
        - The code should be executable and produce the answer
        - Output ONLY the Python code, no explanations or markdown
        - The code should print the final result
        
        Your output MUST be in this exact format:
        <PYTHON_CODE>
        [Your Python code here - must be executable]
        </PYTHON_CODE>"""
    
    user_prompt = f"""Problem: {problem_text}

        Generate Python code to solve this problem. The code should be complete and executable.
        Output the result using print().
        
        Format your response as:
        <PYTHON_CODE>
        [Your complete Python code here]
        </PYTHON_CODE>"""
    
    assistant_start = "<PYTHON_CODE>\n"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_start}
    ]
    
    try:
        # Generate Python code
        response = call_llm(messages, model=selected_model, max_tokens=500)
        full_response = assistant_start + response
        
        # Parse Python code from response
        code_match = re.search(r'<PYTHON_CODE>(.*?)</PYTHON_CODE>', full_response, re.DOTALL)
        
        if not code_match:
            print(f"   ⚠️  No <PYTHON_CODE> tag found, trying to extract code from response")
            # Try to extract code that might be wrapped in other tags or plain
            code_match = re.search(r'```python\s*(.*?)```', full_response, re.DOTALL)
            if not code_match:
                code_match = re.search(r'```\s*(.*?)```', full_response, re.DOTALL)
            if not code_match:
                # Use the response directly as code (after removing assistant_start if present)
                python_code = full_response.replace(assistant_start, "").strip()
            else:
                python_code = code_match.group(1).strip()
        else:
            python_code = code_match.group(1).strip()
        
        print(f"   💻 Generated Python code: {python_code[:100]}...")
        
        # Execute Python code with timeout
        execution_result = None
        execution_error = None
        timeout_seconds = 10  # 10 second timeout
        
        try:
            result = subprocess.run(
                [sys.executable, "-c", python_code],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=None  # Current directory
            )
            
            if result.returncode == 0:
                execution_result = result.stdout.strip()
                print(f"   ✅ Execution successful: {execution_result[:100]}...")
            else:
                execution_error = result.stderr.strip() if result.stderr else "Unknown execution error"
                print(f"   ❌ Execution failed: {execution_error[:100]}...")
        
        except subprocess.TimeoutExpired:
            execution_error = f"Execution timed out after {timeout_seconds} seconds"
            print(f"   ⏱️  {execution_error}")
        except Exception as e:
            execution_error = f"Execution error: {str(e)}"
            print(f"   ❌ {execution_error}")
        
        # Store result in structured format
        if execution_result:
            structured_output = f"<PYTHON_OUTPUT>{execution_result}</PYTHON_OUTPUT>"
            _store_knowledge(state, node_id, scope_id, structured_output)
            return {}
        elif execution_error:
            # Store error message so downstream nodes know Python execution failed
            structured_output = f"<PYTHON_OUTPUT>ERROR: {execution_error}</PYTHON_OUTPUT>"
            _store_knowledge(state, node_id, scope_id, structured_output)
            return {}
        else:
            structured_output = f"<PYTHON_OUTPUT>ERROR: Could not extract or execute Python code</PYTHON_OUTPUT>"
            _store_knowledge(state, node_id, scope_id, structured_output)
            return {}
    
    except Exception as e:
        print(f"   ❌ Error generating Python code: {e}")
        error_output = f"<PYTHON_OUTPUT>ERROR: {str(e)}</PYTHON_OUTPUT>"
        _store_knowledge(state, node_id, scope_id, error_output)
        return {}


def decompose_node(state: AgentState) -> dict:
    """Decompose node - decomposes problem into subproblems, each child solves a different part"""
    node_id = state.get("node_id")
    scope_id = _get_node_scope(state, node_id)
    incoming_knowledge = _get_incoming_knowledge(state, node_id, scope_id)
    
    # Get the problem to decompose
    problem_text = state['problem'][0] if isinstance(state['problem'], list) and state['problem'] else str(state['problem'])
    if incoming_knowledge:
        for inc_id, inc_data in incoming_knowledge.items():
            problem_text = str(inc_data) if not problem_text else problem_text
            break
    
    # Get graph structure to find outgoing nodes
    graph = state.get("graph_structure")
    if not graph:
        print(f"\n📋 Decompose-{node_id}: No graph structure available")
        return {}
    
    outgoing_nodes = graph.get_outgoing_nodes(node_id)
    
    print(f"\n📋 Decompose-{node_id}: Decomposing problem into {len(outgoing_nodes)} subproblems")
    print(f"   📍 Scope: {scope_id}")
    if incoming_knowledge:
        print(f"   📥 Knowledge from: {list(incoming_knowledge.keys())}")
    
    # Decompose the problem into subproblems using LLM
    selected_model = select_model_deterministic(problem_text, EXTRACT_TOPIC_MODELS)
    
    system_prompt = """You are an expert at breaking down complex problems into smaller, independent subproblems.
        Your task is to decompose a problem into N distinct subproblems, where each subproblem:
        - Is a self-contained part of the original problem
        - Can be solved independently
        - Contributes to solving the overall problem when combined
        
        Output the decomposition in a structured format."""
    
    user_prompt = f"""Original Problem: {problem_text}

        Break this problem into {len(outgoing_nodes)} distinct subproblems.
        Each subproblem should be a specific, independent task that contributes to solving the overall problem.
        
        Format:
        <DECOMPOSITION>
        SUBPROBLEM_1: [First subproblem description]
        SUBPROBLEM_2: [Second subproblem description]
        ...
        SUBPROBLEM_{len(outgoing_nodes)}: [Last subproblem description]
        </DECOMPOSITION>"""
    
    assistant_start = "<DECOMPOSITION>\n"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_start}
    ]
    
    try:
        response = call_llm(messages, model=selected_model, max_tokens=800)
        decomposition_response = assistant_start + response
        
        # Parse subproblems
        subproblems = []
        for i in range(1, len(outgoing_nodes) + 1):
            pattern = rf'SUBPROBLEM_{i}:\s*(.+?)(?=SUBPROBLEM_|</DECOMPOSITION>|$)'
            match = re.search(pattern, decomposition_response, re.DOTALL | re.IGNORECASE)
            if match:
                subproblems.append(match.group(1).strip())
            else:
                # Fallback: create generic subproblem
                subproblems.append(f"Subproblem {i}: Part {i} of the original problem")
        
        # Create new scopes for each subproblem (similar to split, but with different semantics)
        scope_mapping = state.get("scope_mapping", {})
        scoped_knowledge = state.get("scoped_knowledge", {})
        
        new_scopes = {}
        for i, target_node_id in enumerate(outgoing_nodes):
            new_scope_id = f"{scope_id}_decomp_{i+1}"
            new_scopes[target_node_id] = new_scope_id
            scope_mapping[target_node_id] = new_scope_id
            
            # Initialize knowledge for new scope with the specific subproblem
            if new_scope_id not in scoped_knowledge:
                from evaluation.agent_state import ScopedKnowledge
                scoped_knowledge[new_scope_id] = ScopedKnowledge(scope_id=new_scope_id)
            
            # Store the subproblem in the scope (nodes in this scope will see it)
            scoped_knowledge[new_scope_id].add(node_id, subproblems[i] if i < len(subproblems) else f"Subproblem {i+1}")
        
        state["scope_mapping"] = scope_mapping
        state["scoped_knowledge"] = scoped_knowledge
        
        print(f"   🆕 Created {len(new_scopes)} subproblem scopes:")
        for target_id, scope in new_scopes.items():
            subproblem = subproblems[list(new_scopes.keys()).index(target_id)] if list(new_scopes.keys()).index(target_id) < len(subproblems) else "Unknown"
            print(f"      Node {target_id} → scope '{scope}': {subproblem[:60]}...")
        
        # Store the full decomposition in the current scope
        _store_knowledge(state, node_id, scope_id, decomposition_response)
        
        return {}
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        # Fallback: create simple subproblems
        scope_mapping = state.get("scope_mapping", {})
        scoped_knowledge = state.get("scoped_knowledge", {})
        
        for i, target_node_id in enumerate(outgoing_nodes):
            new_scope_id = f"{scope_id}_decomp_{i+1}"
            scope_mapping[target_node_id] = new_scope_id
            if new_scope_id not in scoped_knowledge:
                from evaluation.agent_state import ScopedKnowledge
                scoped_knowledge[new_scope_id] = ScopedKnowledge(scope_id=new_scope_id)
            scoped_knowledge[new_scope_id].add(node_id, f"Subproblem {i+1}: {problem_text}")
        
        state["scope_mapping"] = scope_mapping
        state["scoped_knowledge"] = scoped_knowledge
        _store_knowledge(state, node_id, scope_id, f"Error decomposing: {str(e)}")
        
        return {}


def explain_node(state: AgentState) -> dict:
    """Explain node - provides clear explanation of the solution or process.
    
    This node:
    1. Gets the solution/result from incoming knowledge
    2. Generates a clear explanation using LLM
    3. Saves explanation in structured format
    """
    node_id = state.get("node_id")
    scope_id = _get_node_scope(state, node_id)
    incoming_knowledge = _get_incoming_knowledge(state, node_id, scope_id)
    
    # Get the problem to explain
    problem_text = state['problem'][0] if isinstance(state['problem'], list) and state['problem'] else str(state['problem'])
    
    # Get content to explain (from incoming nodes)
    content_to_explain = ""
    if incoming_knowledge:
        for inc_id, inc_data in incoming_knowledge.items():
            content_to_explain = str(inc_data)
            break
    
    if not content_to_explain:
        content_to_explain = problem_text
    
    # Select model deterministically based on problem
    selected_model = select_model_deterministic(problem_text, SOLVER_MODELS)  # Use solver models for explanations
    print(f"\n📖 Explain-{node_id}: Generating explanation (model: {selected_model})")
    print(f"   📍 Scope: {scope_id}")
    if incoming_knowledge:
        print(f"   📥 Knowledge from: {list(incoming_knowledge.keys())}")
    
    # EXPLAIN SPECIFIC PROMPT
    system_prompt = """You are an expert teacher and communicator. Your task is to:
        - Take a solution or result and provide a clear, concise explanation
        - Explain the reasoning, steps, or approach
        - Make it easy to understand for someone learning
        - Keep explanations focused and informative
        
        Output MUST be in this exact format:
        <EXPLANATION>
        [Your clear explanation here - 2-5 sentences]
        </EXPLANATION>"""
    
    user_prompt = f"""Problem: {problem_text}

        Content to explain:
        {content_to_explain[:500]}
        
        Provide a clear explanation of this solution or result:
        <EXPLANATION>
        [Your explanation here]
        </EXPLANATION>"""
    
    assistant_start = "<EXPLANATION>\n"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_start}
    ]
    
    try:
        response = call_llm(messages, model=selected_model, max_tokens=300)
        final_response = assistant_start + response
        
        # Parse explanation
        explanation_match = re.search(r'<EXPLANATION>(.*?)</EXPLANATION>', final_response, re.DOTALL)
        
        if explanation_match:
            explanation = explanation_match.group(1).strip()
            print(f"   ✓ Generated explanation: {explanation[:80]}...")
            
            # Store parsed explanation in structured format
            structured_output = f"<EXPLANATION_OUTPUT>{explanation}</EXPLANATION_OUTPUT>"
            _store_knowledge(state, node_id, scope_id, structured_output)
        else:
            print(f"   ⚠️  No <EXPLANATION> tag found, storing raw response")
            structured_output = f"<EXPLANATION_OUTPUT>{final_response}</EXPLANATION_OUTPUT>"
            _store_knowledge(state, node_id, scope_id, structured_output)
        
        return {}
    except Exception as e:
        print(f"   ❌ Error: {e}")
        error_output = f"<EXPLANATION_OUTPUT>ERROR: {str(e)}</EXPLANATION_OUTPUT>"
        _store_knowledge(state, node_id, scope_id, error_output)
        return {}


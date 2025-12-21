"""Test file for scoped knowledge system.

This file demonstrates how knowledge flows through the graph with scope isolation.
It shows clearly what scope each node has and what knowledge it receives.
"""

from typing import Dict, Any, List, Tuple, Optional
from evaluation.agent_state import AgentState, GlobalKnowledge, GraphStructure, ScopedKnowledge
from evaluation.graph_builder import build_langgraph
from evaluation.agent_nodes import (
    solver_node,
    extract_topic_node,
    validator_node,
    combine_all_node,
    split_node,
    python_solver_node,
    decompose_node,
    explain_node,
)


# Mock LLM call function for testing (returns simulated responses)
def mock_call_llm(messages: List[Dict], model: str = None, max_tokens: int = 128, max_retries: int = 5) -> str:
    """Mock LLM call that returns simulated responses based on node type."""
    user_content = messages[1]["content"] if len(messages) > 1 else ""
    system_content = messages[0]["content"] if len(messages) > 0 else ""
    
    # Print prompt information (full prompts, not truncated)
    print(f"\n💬 PROMPT SENT TO LLM:")
    print(f"   Model: {model}")
    print(f"   System: {system_content}")
    print(f"   User: {user_content}")
    
    # Extract node info from the prompt and return simulated response
    # Note: The response should NOT include the opening tag (e.g., "<SOLUTION>")
    # because the node functions add assistant_start before calling the LLM
    
    if "<SOLUTION>" in user_content:
        # Solver node - simulate response (just the content, not the tags)
        response = "42</SOLUTION>"
        print(f"   ✅ Simulated Response: {response}")
        return response
    elif "<TOPIC_TREE>" in user_content:
        # Extract topic node
        response = """MAIN_TOPIC: Mathematical Problem Solving
├─ SUBTOPIC_1: Arithmetic Operations
└─ SUBTOPIC_2: Problem Decomposition
</TOPIC_TREE>"""
        print(f"   ✅ Simulated Response: {response}")
        return response
    elif "<VALIDATION>" in user_content:
        # Validator node
        response = """RESULT: TRUE
REASONING: The solution is correct
</VALIDATION>"""
        print(f"   ✅ Simulated Response: {response}")
        return response
    elif "<SYNTHESIS>" in user_content:
        # Combine_all node
        response = """FINAL_VERDICT: Solutions from all branches are valid
CONFIDENCE: HIGH
KEY_FINDINGS: Combined results from multiple solvers
</SYNTHESIS>"""
        print(f"   ✅ Simulated Response: {response}")
        return response
    elif "<DECOMPOSITION>" in user_content:
        # Decompose node
        response = """SUBPROBLEM_1: Calculate the sum of 8 and 4
SUBPROBLEM_2: Calculate the product of 8 and 4
</DECOMPOSITION>"""
        print(f"   ✅ Simulated Response: {response}")
        return response
    else:
        response = "Mock response"
        print(f"   ✅ Simulated Response: {response}")
        return response


# Monkey-patch call_llm to use mock
import evaluation.agent_nodes as agent_nodes_module
original_call_llm = agent_nodes_module.call_llm
agent_nodes_module.call_llm = mock_call_llm


def print_node_execution_info(state: AgentState, node_id: int, node_type: str):
    """Print detailed information about node execution."""
    print("\n" + "="*80)
    print(f"🔹 NODE EXECUTION: {node_type}-{node_id}")
    print("="*80)
    
    # Get scope information
    scope_mapping = state.get("scope_mapping", {})
    current_scope = state.get("current_scope", "root")
    node_scope = scope_mapping.get(node_id, current_scope)
    
    print(f"📍 Scope: {node_scope}")
    
    # Get graph structure
    graph = state.get("graph_structure")
    if graph:
        incoming = graph.get_incoming_nodes(node_id)
        outgoing = graph.get_outgoing_nodes(node_id)
        
        print(f"📥 Incoming nodes: {incoming}")
        print(f"📤 Outgoing nodes: {outgoing}")
        
        # Show knowledge from incoming nodes
        scoped_knowledge = state.get("scoped_knowledge", {})
        scope_knowledge = scoped_knowledge.get(node_scope)
        
        # Also check other scopes for combine_all nodes
        all_knowledge = {}
        if scope_knowledge and incoming:
            for inc_id in incoming:
                data = scope_knowledge.get(inc_id)
                if data:
                    all_knowledge[inc_id] = (node_scope, data)
        
        # For combine_all, also search other scopes
        if node_type == "Combine_all" and incoming:
            for scope_id, scope_k in scoped_knowledge.items():
                if scope_id != node_scope:
                    for inc_id in incoming:
                        if inc_id not in all_knowledge:
                            data = scope_k.get(inc_id)
                            if data:
                                all_knowledge[inc_id] = (scope_id, data)
        
        if all_knowledge:
            print(f"\n📚 Knowledge available to {node_type}-{node_id}:")
            for inc_id, (scope, data) in all_knowledge.items():
                inc_type = graph.get_node_type(inc_id) if graph else "unknown"
                print(f"   • {inc_type}-{inc_id} (from scope '{scope}'):")
                print(f"     {str(data)}")
        elif incoming:
            print(f"\n📚 Knowledge from incoming nodes in scope '{node_scope}':")
            for inc_id in incoming:
                inc_type = graph.get_node_type(inc_id) if graph else "unknown"
                print(f"   • {inc_type}-{inc_id}: (no knowledge available)")
    
    # Show what problem text will be used (check if we have incoming knowledge that might override it)
    problem_text = state['problem'][0] if isinstance(state['problem'], list) and state['problem'] else str(state['problem'])
    
    # Get incoming knowledge to check for decomposed/split problems
    scoped_knowledge = state.get("scoped_knowledge", {})
    scope_knowledge = scoped_knowledge.get(node_scope)
    
    if graph and scope_knowledge and incoming:
        for inc_id in incoming:
            inc_data = scope_knowledge.get(inc_id)
            if inc_data:
                inc_type = graph.get_node_type(inc_id)
                if inc_type in ["Decompose", "Split"]:
                    problem_text = f"{str(inc_data)} (from {inc_type}-{inc_id})"
                    break
    
    print(f"\n📝 Problem text that will be used: {problem_text}")
    
    print("="*80)


def print_prompt_info(messages: List[Dict], node_type: str):
    """Print information about the prompt being sent to the LLM."""
    print(f"\n💬 PROMPT for {node_type}:")
    system_prompt = messages[0]["content"] if len(messages) > 0 else ""
    user_prompt = messages[1]["content"] if len(messages) > 1 else ""
    
    print(f"   System: {system_prompt[:100]}...")
    print(f"   User: {user_prompt[:200]}...")


def print_final_knowledge_state(state: AgentState):
    """Print the final state of all knowledge across all scopes."""
    print("\n" + "="*80)
    print("📊 FINAL KNOWLEDGE STATE")
    print("="*80)
    
    scoped_knowledge = state.get("scoped_knowledge", {})
    graph = state.get("graph_structure")
    
    for scope_id, scope_knowledge in scoped_knowledge.items():
        print(f"\n🎯 Scope: {scope_id}")
        if scope_knowledge.node_data:
            for node_id, data in scope_knowledge.node_data.items():
                node_type = graph.get_node_type(node_id) if graph else "unknown"
                print(f"   • {node_type}-{node_id}:")
                print(f"     {str(data)}")
        else:
            print("   (empty)")
    
    print("="*80)


# Wrapper functions to add logging
def logged_solver_node(state: AgentState) -> dict:
    node_id = state.get("node_id")
    node_type = "Solver"
    print_node_execution_info(state, node_id, node_type)
    result = solver_node(state)
    print(f"✅ Solver-{node_id} completed")
    return result


def logged_extract_topic_node(state: AgentState) -> dict:
    node_id = state.get("node_id")
    node_type = "Extract_topic"
    print_node_execution_info(state, node_id, node_type)
    result = extract_topic_node(state)
    print(f"✅ Extract_topic-{node_id} completed")
    return result


def logged_validator_node(state: AgentState) -> dict:
    node_id = state.get("node_id")
    node_type = "Validator"
    print_node_execution_info(state, node_id, node_type)
    result = validator_node(state)
    print(f"✅ Validator-{node_id} completed")
    return result


def logged_combine_all_node(state: AgentState, combine_all_edges: Dict) -> dict:
    node_id = state.get("node_id")
    node_type = "Combine_all"
    print_node_execution_info(state, node_id, node_type)
    result = combine_all_node(state, combine_all_edges)
    print(f"✅ Combine_all-{node_id} completed")
    return result


def logged_split_node(state: AgentState) -> dict:
    node_id = state.get("node_id")
    node_type = "Split"
    print_node_execution_info(state, node_id, node_type)
    result = split_node(state)
    print(f"✅ Split-{node_id} completed")
    return result


def logged_python_solver_node(state: AgentState) -> dict:
    node_id = state.get("node_id")
    node_type = "Python_solver"
    print_node_execution_info(state, node_id, node_type)
    result = python_solver_node(state)
    print(f"✅ Python_solver-{node_id} completed")
    return result


def logged_decompose_node(state: AgentState) -> dict:
    node_id = state.get("node_id")
    node_type = "Decompose"
    print_node_execution_info(state, node_id, node_type)
    result = decompose_node(state)
    print(f"✅ Decompose-{node_id} completed")
    return result


def logged_explain_node(state: AgentState) -> dict:
    node_id = state.get("node_id")
    node_type = "Explain"
    print_node_execution_info(state, node_id, node_type)
    result = explain_node(state)
    print(f"✅ Explain-{node_id} completed")
    return result


def build_test_graph_with_logging(nodes: List[Tuple[int, str]], edges: List[Tuple[int, int]]):
    """Build graph with logging enabled by wrapping node handlers."""
    from langgraph.graph import StateGraph, START, END
    from evaluation.graph_builder import combine_all_node
    
    # Build graph structure first
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
    
    # Create graph structure
    graph_structure = GraphStructure(
        nodes=nodes,
        edges=edges,
        graph_in=graph_in,
        graph_out=graph_out
    )
    
    builder = StateGraph(AgentState)
    
    # Add nodes with logging wrappers
    node_handlers = {
        "Solver": logged_solver_node,
        "Extract_topic": logged_extract_topic_node,
        "Validator": logged_validator_node,
        "Split": logged_split_node,
        "Python_solver": logged_python_solver_node,
        "Decompose": logged_decompose_node,
        "Explain": logged_explain_node,
    }
    
    for node_id, node_type in nodes:
        if node_type in ["START", "END"]:
            continue
        
        node_name = f"{node_type.lower()}_{node_id}"
        
        if node_type == "Combine_all":
            def make_combine_node(nid):
                def handler(state):
                    state["node_id"] = nid
                    print_node_execution_info(state, nid, "Combine_all")
                    result = combine_all_node(state, combine_all_edges)
                    print(f"✅ Combine_all-{nid} completed")
                    return result
                return handler
            builder.add_node(node_name, make_combine_node(node_id))
        elif node_type in node_handlers:
            def make_typed_node(nid, ntype, handler_func):
                def handler(state):
                    state["node_id"] = nid
                    return handler_func(state)
                return handler
            builder.add_node(node_name, make_typed_node(node_id, node_type, node_handlers[node_type]))
        else:
            # Generic node
            def make_generic_node(nid, ntype):
                def handler(state: AgentState) -> dict:
                    print_node_execution_info(state, nid, ntype)
                    problem_text = state['problem'][0] if isinstance(state['problem'], list) and state['problem'] else str(state['problem'])
                    return {"result": [f"Generic node {nid}: {problem_text[:50]}..."]}
                return handler
            builder.add_node(node_name, make_generic_node(node_id, node_type))
    
    # Add edges
    def get_node_name(node_id: int) -> Optional[str]:
        node_type = id_to_type.get(node_id)
        if node_type in ["START", "END"]:
            return None
        return f"{node_type.lower()}_{node_id}"
    
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
    
    for src, dst in edges:
        src_type = id_to_type.get(src)
        dst_type = id_to_type.get(dst)
        if src_type in ["START", "END"] or dst_type in ["START", "END"]:
            continue
        src_name = get_node_name(src)
        dst_name = get_node_name(dst)
        if src_name and dst_name:
            builder.add_edge(src_name, dst_name)
    
    if end_node_id is not None:
        for src in graph_in.get(end_node_id, []):
            src_name = get_node_name(src)
            if src_name:
                builder.add_edge(src_name, END)
    
    compiled_graph = builder.compile()
    
    # Wrap invoke to ensure graph_structure is set
    original_invoke = compiled_graph.invoke
    def invoke_with_graph_state(input_state):
        if "scoped_knowledge" not in input_state or not input_state.get("scoped_knowledge"):
            input_state["scoped_knowledge"] = {"root": ScopedKnowledge(scope_id="root")}
        if "scope_mapping" not in input_state:
            input_state["scope_mapping"] = {}
        if "current_scope" not in input_state:
            input_state["current_scope"] = "root"
        if "graph_structure" not in input_state or input_state.get("graph_structure") is None:
            input_state["graph_structure"] = graph_structure
        return original_invoke(input_state)
    
    compiled_graph.invoke = invoke_with_graph_state
    return compiled_graph


def main():
    """Run comprehensive test with all node types."""
    
    print("\n" + "="*80)
    print("🧪 TESTING SCOPED KNOWLEDGE SYSTEM")
    print("="*80)
    
    # Example graph showing both Decompose and Split:
    # START → Decompose → [Solver_1, Solver_2] → Combine_all → Split → [Extract_topic, Python_solver] → Combine_all → Validator → END
    # Decompose: Each child solves a DIFFERENT subproblem, all must complete
    # Split: Each child solves the SAME problem, any can be used
    test_graph = {
        "name": "Comprehensive Test Graph (Decompose + Split)",
        "nodes": [
            (0, "START"),
            (1, "Decompose"),
            (2, "Solver"),      # Solves subproblem 1
            (3, "Solver"),      # Solves subproblem 2
            (4, "Combine_all"), # Waits for both decomposed solvers
            (5, "Split"),
            (6, "Extract_topic"),
            (7, "Python_solver"),
            (8, "Combine_all"), # Combines split results
            (9, "Validator"),
            (10, "Explain"),
            (11, "END")
        ],
        "edges": [
            (0, 1),    # START → Decompose
            (1, 2),    # Decompose → Solver_1 (gets subproblem 1)
            (1, 3),    # Decompose → Solver_2 (gets subproblem 2)
            (2, 4),    # Solver_1 → Combine_all
            (3, 4),    # Solver_2 → Combine_all (waits for both)
            (4, 5),    # Combine_all → Split
            (5, 6),    # Split → Extract_topic (same problem as Python_solver)
            (5, 7),    # Split → Python_solver (same problem as Extract_topic)
            (6, 8),    # Extract_topic → Combine_all
            (7, 8),    # Python_solver → Combine_all
            (8, 9),    # Combine_all → Validator
            (9, 10),   # Validator → Explain
            (10, 11),  # Explain → END
        ]
    }
    
    print(f"\n📋 Test Graph: {test_graph['name']}")
    print(f"   Nodes: {len([n for n in test_graph['nodes'] if n[1] not in ['START', 'END']])}")
    print(f"   Edges: {len(test_graph['edges'])}")
    
    # Build graph
    graph = build_test_graph_with_logging(test_graph["nodes"], test_graph["edges"])
    
    # Initial state - a more complex problem that can be decomposed
    problem = "Find the sum and product of 8 and 4, then explain the relationship between addition and multiplication."
    
    initial_state: AgentState = {
        "problem": [problem],
        "global_knowledge": GlobalKnowledge(),
        "graph_structure": None,  # Will be set by graph builder
        "scoped_knowledge": {"root": ScopedKnowledge(scope_id="root")},
        "scope_mapping": {},
        "current_scope": "root",
        "result": [],
        "node_type": None,
        "node_id": None,
        "solution": None
    }
    
    print("\n" + "="*80)
    print("🚀 EXECUTING GRAPH")
    print("="*80)
    print(f"Problem: {problem}\n")
    
    try:
        result = graph.invoke(initial_state)
        
        # Print final state
        print_final_knowledge_state(result)
        
        print("\n" + "="*80)
        print("✅ EXECUTION COMPLETE")
        print("="*80)
        print(f"\nFinal solution: {result.get('solution', 'N/A')}")
        
        # Print knowledge flow summary
        print("\n" + "="*80)
        print("📋 KNOWLEDGE FLOW SUMMARY")
        print("="*80)
        
        scoped_knowledge = result.get("scoped_knowledge", {})
        graph = result.get("graph_structure")
        
        print("\nKnowledge Flow by Scope:")
        for scope_id in sorted(scoped_knowledge.keys()):
            scope_knowledge = scoped_knowledge[scope_id]
            print(f"\n  🎯 Scope: {scope_id}")
            if scope_knowledge.node_data:
                for node_id in sorted(scope_knowledge.node_data.keys()):
                    node_type = graph.get_node_type(node_id) if graph else "unknown"
                    data = scope_knowledge.node_data[node_id]
                    print(f"     {node_type}-{node_id} → {str(data)}")
        
        print("\n" + "="*80)
        
        return result
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()


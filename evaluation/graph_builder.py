"""LangGraph builder functions."""

from typing import List, Optional, Tuple

from langgraph.graph import StateGraph, START, END

from evaluation.agent_state import AgentState
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


def visualize_graph_ascii(graph):
    """Print ASCII representation of the graph"""
    print("\n" + "="*80)
    print("GRAPH STRUCTURE (ASCII)")
    print("="*80)
    try:
        print(graph.get_graph().draw_ascii())
    except Exception as e:
        print(f"Could not generate ASCII visualization: {e}")


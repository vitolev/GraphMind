"""Test file for LangGraph multi-agent LLM system.

This file provides example usage and testing code.
The actual implementation has been split into separate modules:
- agent_state.py: State definitions
- agent_nodes.py: Node functions
- graph_builder.py: Graph building
- llm_callers.py: LLM calling functions
- llm_providers.py: Provider setup
- model_selection.py: Model pools and selection
"""

from typing import Dict, Any

# Import from new modular structure
from evaluation.agent_state import AgentState, GlobalKnowledge
from evaluation.graph_builder import build_langgraph, visualize_graph_ascii
from evaluation.llm_callers import set_llm_provider

# Re-export for backward compatibility
__all__ = [
    'AgentState',
    'GlobalKnowledge',
    'build_langgraph',
    'visualize_graph_ascii',
    'set_llm_provider',
]

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

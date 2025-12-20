import operator
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, TypedDict, Annotated
from groq import Groq
from langgraph.graph import StateGraph, START, END
import time


# ============================================================================
# LLM SETUP - GROQ FREE VERSION
# ============================================================================

client = Groq()


def call_groq_with_retry(messages: List[Dict], model: str = "llama-3.1-8b-instant", max_tokens: int = 128, max_retries: int = 5) -> str:
    """Generic Groq API call with retry logic"""
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
                wait_time = retry_delay * (2 ** attempt)
                print(f"      ⏳ Rate limited. Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
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
    print(f"\n🔧 Solver-{node_id}: Quick solution (max 100 chars)")
    
    problem_text = state['problem'][0] if isinstance(state['problem'], list) and state['problem'] else str(state['problem'])
    
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
        response = call_groq_with_retry(messages, model="llama-3.1-8b-instant", max_tokens=150)
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
    print(f"\n📖 Extract_topic-{node_id}: Building topic tree (hiring guide)")
    
    problem_text = state['problem'][0] if isinstance(state['problem'], list) and state['problem'] else str(state['problem'])
    
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
        response = call_groq_with_retry(messages, model="llama-3.1-8b-instant", max_tokens=800)
        final_response = assistant_start + response
        state['global_knowledge'].add(node_id, final_response)
        print(f"   ✓ Response: {final_response[:80]}...")
        return {"result": [final_response]}
    except Exception as e:
        print(f"   ❌ Error: {e}")
        state['global_knowledge'].add(node_id, f"Error: {str(e)}")
        return {"result": [f"Error: {str(e)}"]}


def validator_node(state: AgentState) -> dict:
    """Validator node - critical analysis, looks for counter-examples"""
    node_id = state.get("node_id")
    print(f"\n✓ Validator-{node_id}: Critical validation (finding counter-examples)")
    
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
        response = call_groq_with_retry(messages, model="llama-3.1-8b-instant", max_tokens=400)
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
    print(f"\n🔗 Combine_all-{node_id}: Synthesizing all results")
    
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
        response = call_groq_with_retry(messages, model="llama-3.1-8b-instant", max_tokens=500)
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
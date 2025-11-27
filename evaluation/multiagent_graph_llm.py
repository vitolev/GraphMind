import operator
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, TypedDict, Annotated, Literal
from groq import Groq
from langgraph.graph import StateGraph, START, END

# ============================================================================
# SYSTEM PROMPTS FOR EACH NODE TYPE
# ============================================================================

MESSAGE_TEMPLATES = {
       "Solver": {
        "system": """You are an expert problem solver. Your task is to
        read the task carefully and provide just the answer to the problem
        that will be evaluated based on only the corrected expression or number
        or whatever the problem requires as an answer.""",
        
        "user": """Problem: {problem}

        Your answer should be in the following format, the answers with more than 100 characters will be evaluated as incorrect:
        <SOLUTION>
        [Your detailed solution here]
        </SOLUTION>""",
        "assistant_start": "The final solution to this problem is: <SOLUTION> "
    },

    "Solver-slow": {
        "system": """You are an expert problem solver. Your task is to:
1. Understand the problem clearly
2. Break it down into manageable steps
3. Provide a clear, logical solution
4. Explain your reasoning

Be concise but thorough. Focus on practical solutions.""",
        
        "user": """Problem: {problem}

Your answer should be in the following format:
<SOLUTION>
[Your detailed solution here]
</SOLUTION>""",
        
        "assistant_start": "The solution to this problem is:\n<SOLUTION>\n"
    },

    "Extract_topic": {
        "system": """You are an expert at identifying key information. Your task is to:
1. Read the input carefully
2. Extract the main topic or central theme
3. Identify 2-3 key subtopics
4. Summarize the essence in 1-2 sentences

Be precise and focus on the most important information.""",
        
        "user": """Content: {problem}

Extract the main topic in the following format:
<TOPIC>
[Main topic here]
</TOPIC>

<SUBTOPICS>
[Subtopics here]
</SUBTOPICS>""",
        
        "assistant_start": "The main topic is:\n<TOPIC>\n"
    },

    "Validator": {
        "system": """You are a quality assurance expert. Your task is to:
1. Review the provided information
2. Check for logical consistency
3. Identify any gaps or issues
4. Provide a validation score (0-100)
5. Suggest improvements if needed

Be fair but thorough in your assessment.""",
        
        "user": """Information to validate: {problem}

Provide validation in the following format:
<VALIDATION>
SCORE: [0-100]
STATUS: [VALID/INVALID]
ISSUES: [List any issues or "NONE"]
IMPROVEMENTS: [Suggestions or "NONE"]
</VALIDATION>""",
        
        "assistant_start": "Validation result:\n<VALIDATION>\n"
    },

    "Combine_all": {
        "system": """You are an expert synthesizer. Your task is to:
1. Review all the provided inputs
2. Find common themes and connections
3. Merge insights without losing important details
4. Create a cohesive summary
5. Highlight the most valuable insights

Be integrative and create meaningful connections.""",
        
        "user": """Review and combine these results:
{problem}

Provide a synthesis in the following format:
<SYNTHESIS>
[Your combined analysis here]
</SYNTHESIS>

<KEY_INSIGHTS>
[Most valuable insights]
</KEY_INSIGHTS>""",
        
        "assistant_start": "Combined synthesis:\n<SYNTHESIS>\n"
    },

    "Python_executor": {
        "system": """You are an expert Python programmer. Your task is to:
1. Understand the problem completely
2. Write a complete, executable Python program to solve it
3. Include all necessary imports and error handling
4. Ensure the program produces the solution as output
5. The program should be ready to run immediately

Write ONLY the Python code, nothing else. No markdown, no explanations.""",
        
        "user": """Problem: {problem}

Write a Python program to solve this. The program should print the solution.""",
        
        "assistant_start": None  # Python executor doesn't need pre-filled assistant
    },

    "Default": {
        "system": """You are a helpful assistant. Your task is to:
1. Understand the request
2. Provide a thoughtful response
3. Be clear and concise
4. Add relevant context
5. Offer next steps if applicable

Be helpful and accurate.""",
        
        "user": "{problem}",
        
        "assistant_start": "Here is my response:\n"
    }
}

# ============================================================================
# LLM SETUP - GROQ FREE VERSION
# ============================================================================

client = Groq()

def generate_response(message: str, node_type: str = "Default", model: str = "llama-3.1-8b-instant", use_assistant_start = True) -> str:

    template = MESSAGE_TEMPLATES.get(node_type, MESSAGE_TEMPLATES["Default"])
    system_prompt = template["system"]
    
    # Format the user message from template
    user_message = template["user"].format(problem=message)
    
    print(f"   🧠 Groq LLM call")
    print(f"      Node Type: {node_type}")
    print(f"      Message: {message[:100]}...")
    
    import time

    max_retries = 5
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Add assistant start if specified and available
            if use_assistant_start and template.get("assistant_start"):
                messages.append({
                    "role": "assistant",
                    "content": template["assistant_start"]
                })
            
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model,
                max_completion_tokens=128,
            )
            
            response = chat_completion.choices[0].message.content
            
            # If we used assistant_start, prepend it to the response
            if use_assistant_start and template.get("assistant_start"):
                response = template["assistant_start"] + response
            
            return response
        except Exception as e:
            print(f"   ❌ Groq LLM call failed on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                print(f"      Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("      Max retries reached. Raising exception.")
                raise

# ============================================================================
# STATE DEFINITIONS - WITH REDUCERS
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
    node_type: Optional[str]  # Current node type for routing
    node_id: Optional[int]    # Current node ID for routing

# ============================================================================
# NODE FUNCTIONS - PURE LOGIC SEPARATED
# ============================================================================

def solver_node(state: AgentState) -> dict:
    node_id = state.get("node_id")
    print(f"\n🔧 Solver-{node_id}: Using Groq LLM")
    problem_text = state['problem'][0] if isinstance(state['problem'], list) and state['problem'] else str(state['problem'])
    prompt = f"Solve this problem: {problem_text}"
    response = generate_response(prompt, model="moonshotai/kimi-k2-instruct-0905" ,node_type="Solver")
    state['global_knowledge'].add(node_id, response)
    print(f"   Response: {response}...")
    return {"result": [response]}


def extract_topic_node(state: AgentState) -> dict:
    node_id = state.get("node_id")
    print(f"\n📖 Extract_topic-{node_id}: Using Groq LLM")
    problem_text = state['problem'][0] if isinstance(state['problem'], list) and state['problem'] else str(state['problem'])
    prompt = f"Extract the main topic from: {problem_text}"
    response = generate_response(prompt, model="meta-llama/llama-4-maverick-17b-128e-instruct")
    state['global_knowledge'].add(node_id, response)
    print(f"   Response: {response[:100]}...")
    return {"result": [response]}


def validator_node(state: AgentState) -> dict:
    node_id = state.get("node_id")
    print(f"\n✓ Validator-{node_id}: Validating")
    is_valid = True
    state['global_knowledge'].add(node_id, {"valid": is_valid})
    return {"result": [{"valid": is_valid}]}


def combine_all_node(state: AgentState, combine_all_edges: dict) -> dict:
    node_id = state.get("node_id")
    print(f"\n🔗 Combine_all-{node_id}: Merging with Groq LLM")
    incoming = combine_all_edges.get(node_id, [])
    
    combined_data = []
    for inc_id in incoming:
        data = state['global_knowledge'].get(inc_id)
        if data:
            combined_data.append(str(data)[:500])  # Limit size for prompt
    
    combined_text = " ".join(combined_data)
    prompt = f"Combine and summarize: {combined_text}"
    response = generate_response(prompt, model="openai/gpt-oss-120b")
    state['global_knowledge'].add(node_id, response)
    print(f"   Combined: {response[:500]}...")
    return {"result": [response]}

# ============================================================================
# ROUTING FUNCTIONS - FOR CONDITIONAL EDGES
# ============================================================================

def route_by_node_type(state: AgentState, id_to_type: dict) -> str:
    """Route to appropriate node based on type"""
    node_id = state.get("node_id")
    node_type = id_to_type.get(node_id)
    
    if node_type == "Solver":
        return "solver"
    elif node_type == "Python_solver":
        return "python_solver"
    elif node_type == "Extract_topic":
        return "extract_topic"
    elif node_type == "Validator":
        return "validator"
    elif node_type == "Combine_all":
        return "combine_all"
    elif node_type == "True_pass":
        return "true_pass"
    elif node_type == "False_pass":
        return "false_pass"
    else:
        return END

# ============================================================================
# BUILD LANGGRAPH FROM STRUCTURE
# ============================================================================

def build_langgraph(nodes: List[Tuple[int, str]], edges: List[Tuple[int, int]]):
    """Build LangGraph using LangGraph's native routing - handles custom node types"""
    
    # Build maps
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
    
    # Process special nodes
    combine_all_edges = {}
    for node_id, node_type in nodes:
        if node_type == "Combine_all":
            incoming = graph_in.get(node_id, [])
            combine_all_edges[node_id] = incoming
    
    print("🔨 Building LangGraph (Native Routing)...")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Edges: {len(edges)}")
    print(f"  Combine_all edges: {combine_all_edges}")
    
    # Create builder
    builder = StateGraph(AgentState)
    
    # Add execution nodes (skip START and END)
    print("\n  Adding nodes:")
    
    for node_id, node_type in nodes:
        if node_type in ["START", "END"]:
            continue  # Skip START and END - they're implicit
        
        if node_type == "Solver":
            builder.add_node(f"solver_{node_id}", solver_node)
            print(f"    ✓ solver_{node_id}")
        
        elif node_type == "Extract_topic":
            builder.add_node(f"extract_topic_{node_id}", extract_topic_node)
            print(f"    ✓ extract_topic_{node_id}")
        
        elif node_type == "Validator":
            builder.add_node(f"validator_{node_id}", validator_node)
            print(f"    ✓ validator_{node_id}")
        
        elif node_type == "Combine_all":
            def make_combine_node(nid):
                def handler(state):
                    state["node_id"] = nid
                    return combine_all_node(state, combine_all_edges)
                return handler
            builder.add_node(f"combine_all_{node_id}", make_combine_node(node_id))
            print(f"    ✓ combine_all_{node_id}")
        
        elif node_type in ["True_pass", "False_pass"]:
            def make_pass_node(nid, ntype):
                def handler(state):
                    print(f"\n{'✓' if ntype == 'True_pass' else '✗'} {ntype}-{nid}")
                    return {"result": []}
                return handler
            builder.add_node(f"{node_type.lower()}_{node_id}", make_pass_node(node_id, node_type))
            print(f"    ✓ {node_type.lower()}_{node_id}")
        
        else:
            # DEFAULT: Handle any unknown node type as a generic solver
            print(f"    ℹ️  Unknown node type '{node_type}', treating as generic solver")
            
            def make_generic_node(nid, ntype):
                def handler(state: AgentState) -> dict:
                    print(f"\n🔧 {ntype}-{nid}: Generic solver")
                    problem_text = state['problem'][0] if isinstance(state['problem'], list) and state['problem'] else str(state['problem'])
                    prompt = f"Solve this problem: {problem_text}"
                    response = generate_response(prompt)
                    state['global_knowledge'].add(nid, response)
                    print(f"   Response: {response[:100]}...")
                    return {"result": [response]}
                return handler
            
            builder.add_node(f"{node_type.lower()}_{node_id}", make_generic_node(node_id, node_type))
            print(f"    ✓ {node_type.lower()}_{node_id}")
    
    # Helper function to get node name
    def get_node_name(node_id: int) -> Optional[str]:
        """Get the full node name from node ID"""
        node_type = id_to_type.get(node_id)
        
        if node_type in ["START", "END"]:
            return None
        
        if node_type == "Solver":
            return f"solver_{node_id}"
        elif node_type == "Extract_topic":
            return f"extract_topic_{node_id}"
        elif node_type == "Validator":
            return f"validator_{node_id}"
        elif node_type == "Combine_all":
            return f"combine_all_{node_id}"
        elif node_type == "True_pass":
            return f"true_pass_{node_id}"
        elif node_type == "False_pass":
            return f"false_pass_{node_id}"
        else:
            # Return generic name for unknown types
            return f"{node_type.lower()}_{node_id}"
    
    # Add edges
    print("\n  Adding edges:")
    
    # Find START and END node IDs
    start_node_id = None
    end_node_id = None
    for node_id, node_type in nodes:
        if node_type == "START":
            start_node_id = node_id
        elif node_type == "END":
            end_node_id = node_id
    
    # Connect START to nodes that have incoming edges from START
    if start_node_id is not None:
        for dst in graph_out.get(start_node_id, []):
            dst_name = get_node_name(dst)
            if dst_name:
                builder.add_edge(START, dst_name)
                print(f"    ✓ START → {dst_name}")
    
    # Connect regular edges
    for src, dst in edges:
        src_type = id_to_type.get(src)
        dst_type = id_to_type.get(dst)
        
        # Skip edges involving START or END
        if src_type in ["START", "END"] or dst_type in ["START", "END"]:
            continue
        
        src_name = get_node_name(src)
        dst_name = get_node_name(dst)
        
        if src_name and dst_name:
            builder.add_edge(src_name, dst_name)
            print(f"    ✓ {src_name} → {dst_name}")
    
    # Connect terminal nodes to END
    if end_node_id is not None:
        for src in graph_in.get(end_node_id, []):
            src_name = get_node_name(src)
            if src_name:
                builder.add_edge(src_name, END)
                print(f"    ✓ {src_name} → END")
    
    print("\n✓ LangGraph compilation complete")
    return builder.compile()


# ============================================================================
# VISUALIZATION FUNCTIONS - LANGGRAPH
# ============================================================================

import os
from typing import Optional
from pathlib import Path


def visualize_graph_ascii(graph):
    """Print ASCII representation of the graph (terminal-friendly)"""
    print("\n" + "="*80)
    print("GRAPH STRUCTURE (ASCII)")
    print("="*80)
    print(graph.get_graph().draw_ascii())


def visualize_graph_mermaid(graph):
    """Print Mermaid diagram code (can be pasted into mermaid.live)"""
    print("\n" + "="*80)
    print("GRAPH STRUCTURE (MERMAID CODE)")
    print("="*80)
    mermaid_code = graph.get_graph().draw_mermaid()
    print(mermaid_code)
    return mermaid_code


def save_graph_mermaid(graph, filename: str = "langgraph_diagram.md"):
    """Save Mermaid diagram code to file"""
    mermaid_code = graph.get_graph().draw_mermaid()
    with open(filename, 'w') as f:
        f.write(mermaid_code)
    print(f"✓ Mermaid code saved to: {filename}")
    print(f"  View at: https://mermaid.live/")


def visualize_graph_png(graph, filename: str = "langgraph_diagram.png", method: str = "api"):
    """
    Save graph as PNG image
    
    Args:
        graph: Compiled LangGraph graph
        filename: Output filename
        method: "api" (mermaid.ink), "pyppeteer" (local browser), or "graphviz"
    """
    try:
        if method == "graphviz":
            # Use Graphviz (requires: pip install graphviz)
            png_data = graph.get_graph().draw_png()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        with open(filename, 'wb') as f:
            f.write(png_data)
        print(f"✓ Graph PNG saved to: {filename}")
        return filename
    
    except ImportError as e:
        print(f"❌ Error: {e}")
        print(f"   Install required package for '{method}' method")
        if method == "pyppeteer":
            print("   pip install pyppeteer")
        elif method == "graphviz":
            print("   pip install graphviz")
        return None


def visualize_graph_full(graph, output_dir: str = "."):
    """
    Generate all graph visualizations
    
    Args:
        graph: Compiled LangGraph graph
        output_dir: Directory to save outputs
    """
    print("\n" + "="*80)
    print("GENERATING ALL VISUALIZATIONS")
    print("="*80)
    
    # Create output directory if needed
    Path(output_dir).mkdir(exist_ok=True)
    
    # 1. ASCII (always works)
    print("\n✓ Generating ASCII visualization...")
    visualize_graph_ascii(graph)
    
    # 2. Mermaid code
    print("\n✓ Generating Mermaid code...")
    mermaid_file = os.path.join(output_dir, "langgraph_diagram.md")
    save_graph_mermaid(graph, mermaid_file)
    
    # 3. Try PNG (mermaid.ink API)
    print("\n✓ Attempting to generate PNG (via mermaid.ink API)...")
    png_file = os.path.join(output_dir, "langgraph_diagram.png")
    result = visualize_graph_png(graph, png_file, method="api")
    if result:
        print(f"  View image: {result}")
    
    print("\n" + "="*80)
    print("✓ VISUALIZATIONS COMPLETE")
    print("="*80)


def print_graph_info(graph):
    """Print detailed graph information"""
    print("\n" + "="*80)
    print("GRAPH INFORMATION")
    print("="*80)
    
    graph_obj = graph.get_graph()
    
    # Get nodes
    nodes = list(graph_obj.nodes)
    print(f"\nNodes ({len(nodes)}):")
    for node in nodes:
        print(f"  - {node}")
    
    # Get edges
    edges = list(graph_obj.edges)
    print(f"\nEdges ({len(edges)}):")
    for src, dst in edges:
        print(f"  - {src} → {dst}")
    
    # Print as Mermaid for reference
    print(f"\nMermaid representation:")
    print(graph_obj.draw_mermaid())


# ============================================================================
# INTEGRATION WITH BUILD_LANGGRAPH
# ============================================================================

def build_langgraph_with_viz(nodes, edges, output_dir: str = "."):
    """
    Build LangGraph and generate visualizations
    
    Args:
        nodes: List of (node_id, node_type) tuples
        edges: List of (src, dst) tuples
        output_dir: Directory to save visualizations
    
    Returns:
        Compiled graph
    """
    # Build graph
    graph = build_langgraph(nodes, edges)
    
    # Generate visualizations
    visualize_graph_full(graph, output_dir)
    
    return graph



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
        (3, "Solver"),
        (4, "Combine_all"),
        (5, "END")
    ],
    "edges": [
        (0, 1),
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 4),
        (4, 5)
    ]
}


EXAMPLE_3 = {
    "name": "Conditional Routing",
    "nodes": [
        (0, "START"),
        (1, "Validator"),
        (2, "True_pass"),
        (3, "False_pass"),
        (4, "Solver"),
        (5, "Solver"),
        (6, "END")
    ],
    "edges": [
        (0, 1),
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
        (5, 6)
    ]
}




# ============================================================================
# MAIN
# ============================================================================


def run_example(example: Dict[str, Any], problem: str):
    """Run example with visualizations"""
    
    print("\n" + "="*80)
    print(f"EXAMPLE: {example['name']}")
    print("="*80)
    
    # Build graph
    graph = build_langgraph(example["nodes"], example["edges"])
    
    # Visualize BEFORE executing
    print("\n📊 VISUALIZING GRAPH STRUCTURE")
    visualize_graph_ascii(graph)  # Always works
    save_graph_mermaid(graph, f"visualizations/{example['name'].lower().replace(' ', '_')}.md")
    
    # Try to save PNG
    try:
        visualize_graph_png(graph, f"visualizations/{example['name'].lower().replace(' ', '_')}.png")
    except Exception as e:
        print(f"  Note: PNG generation skipped ({type(e).__name__})")
    
    # Create initial state
    initial_state: AgentState = {
        "problem": [problem],
        "global_knowledge": GlobalKnowledge(),
        "result": [],
        "node_type": None,
        "node_id": None
    }
    
    # Execute
    print("\n" + "="*80)
    print("EXECUTING LANGGRAPH")
    print("="*80)
    
    try:
        result = graph.invoke(initial_state)
        
        # Print results...
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        
        results = result['result'] if isinstance(result['result'], list) else [result['result']]
        print(f"\nFinal results ({len(results)} items):")
        for i, res in enumerate(results):
            print(f"  [{i}]: {str(res)[:200]}")
        
        print(f"\nAll knowledge entries:")
        for node_id, data in result['global_knowledge'].entries.items():
            print(f"  Node {node_id}: {str(data)[:100]}")
        
        return result
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all examples with visualizations"""
    
    print("\n" + "="*80)
    print("LANGGRAPH WITH GROQ - MULTI-AGENT SYSTEM")
    print("="*80)
    
    # Create visualizations directory
    Path("visualizations").mkdir(exist_ok=True)
    
    # Example 1: Linear
    run_example(EXAMPLE_1, "What is machine learning?")
    
    # Example 2: With Combine (parallel nodes!)
    run_example(EXAMPLE_2, "Explain neural networks")
    
    # Example 3: Conditional (parallel nodes!)
    run_example(EXAMPLE_3, "Solve this complex problem")
    
    print("\n" + "="*80)
    print("✓ ALL EXAMPLES COMPLETED")
    print("="*80)
    
    # Generate summary of all visualizations
    print("\n✓ All visualizations saved to: ./visualizations/")


if __name__ == "__main__":
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
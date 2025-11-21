import random
import networkx as nx
import matplotlib.pyplot as plt

import sys
import os
# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from data_management.graph_storage import Graph

# ======================
#  Agent types and rules
# ======================
NODE_TYPES = [
    "Validator", "True_pass", "False_pass",
    "Split", "Decompose_2", "Decompose_3", "Decompose_4",
    "Combine_all", "Combine_any",
    "Solver", "Python_solver", "Explain", "Extract_topic",
    "START", "END"
]
RULES = {
    "Validator": {"branches": 2, "allowed_children": ["True_pass", "False_pass"]},
    "True_pass": {"branches": 1, "allowed_children": []},
    "False_pass": {"branches": 1, "allowed_children": ["Solver", "Python_solver", "Explain", "Extract_topic", "Decompose_2", "Decompose_3", "Decompose_4", ]},
    
    "Split": {"branches": 2, "allowed_children": ["Solver", "Python_solver", "Explain", "Extract_topic", "Decompose_2", "Decompose_3", "Decompose_4"]},
    "Decompose_2": {"branches": 2, "allowed_children": ["Split", "Solver", "Python_solver", "Explain", "Extract_topic", "Decompose_2", "Decompose_3", "Decompose_4"]},
    "Decompose_3": {"branches": 3, "allowed_children": ["Split", "Solver", "Python_solver", "Explain", "Extract_topic", "Decompose_2", "Decompose_3", "Decompose_4"]},
    "Decompose_4": {"branches": 4, "allowed_children": ["Split", "Solver", "Python_solver", "Explain", "Extract_topic", "Decompose_2", "Decompose_3", "Decompose_4"]},

    "Combine_all": {"branches": 1, "allowed_children": ["Solver", "Python_solver", "Explain", "Extract_topic", "Decompose_2", "Decompose_3", "Decompose_4"]},
    "Combine_any": {"branches": 1, "allowed_children": ["Solver", "Python_solver", "Explain", "Extract_topic", "Decompose_2", "Decompose_3", "Decompose_4", "END"]},

    "Solver": {"branches": 1, "allowed_children": ["Validator", "END"]},
    "Python_solver": {"branches": 1, "allowed_children": ["Validator", "END"]},
    "Explain": {"branches": 1, "allowed_children": ["Solver", "Python_solver", "Extract_topic", "Split"]},
    "Extract_topic": {"branches": 1, "allowed_children": ["Solver", "Python_solver", "Explain", "Split"]},

    "START": {"branches": 1, "allowed_children": ["Split", "Solver", "Python_solver", "Explain", "Extract_topic", "Decompose_2", "Decompose_3", "Decompose_4"]},
    "END": {"branches": 0, "allowed_children": []}
}
DEPTH_INCREASE_NODES = ["Decompose_2", "Decompose_3", "Decompose_4", "Split", "Validator"]

# ======================
#  Node class
# ======================

class Node:
    def __init__(self, type_name: str):
        if type_name not in NODE_TYPES:
            raise ValueError(f"Invalid node type: {type_name}")
        self.type_name = type_name
        self.children = []

    def add_child(self, node):
        self.children.append(node)

    def __repr__(self):
        return f"{self.type_name}()"

# ---------------------------------------
# Random graph builder
# ---------------------------------------
def _build_random_graph(max_depth=2):
    start = Node("START")

    def _pick_random_child(node_type):
        allowed = RULES[node_type]["allowed_children"]
        return random.choice(allowed)
    
    def _pick_no_depth_increase_child(node_type):
        allowed = [child for child in RULES[node_type]["allowed_children"] if child not in DEPTH_INCREASE_NODES]
        return random.choice(allowed)

    def _rec(node, depth, max_depth):
        node_type = node.type_name
        rule = RULES[node_type]
        num_branches = rule["branches"]

        # For multi-branch nodes like Decompose_X or Split
        if node_type in ["Decompose_2", "Decompose_3", "Decompose_4", "Split"]:
            depth += 1
            children_to_combine = []
            for _ in range(num_branches):
                if depth == max_depth:
                    # Allow only nodes that dont lead to further depth increase
                    child_type = _pick_no_depth_increase_child(node_type)
                else:
                    child_type = _pick_random_child(node_type)   # these node_types for sure do not have "END" as allowed child, so no need for explicit check
                child = _rec(Node(child_type), depth, max_depth)
                node.add_child(child)
                children_to_combine.append(child)
                
            # After branches are generated, attach Combine_all to merge them
            depth -= 1
            combine_all_node = Node("Combine_all")
            for child in children_to_combine:
                # Find the last node in each branch to connect to Combine_all
                last_node = child
                while last_node.children:
                    last_node = last_node.children[0]
                last_node.add_child(combine_all_node)

            # Continue building from Combine_all
            _rec(combine_all_node, depth, max_depth)

            # Return the current node
            return node

        # For Validator
        if node_type == "Validator":
            # Two branches: True_pass and False_pass
            true_node = Node("True_pass")   # True branch just passes through
            false_node = _rec(Node("False_pass"), depth + 1, max_depth) # False branch continues with recursive build
            node.add_child(true_node)
            node.add_child(false_node)

            # Combine both branches with Combine_any
            combine_any_node = Node("Combine_any")
            true_node.add_child(combine_any_node)

            # Attach the last node of false branch to Combine_any
            last_false = false_node
            while last_false.children:
                last_false = last_false.children[0]
            last_false.add_child(combine_any_node)

            # Continue building from Combine_any
            _rec(combine_any_node, depth, max_depth)

            # Return the current validator node
            return node

        # For other nodes: False_pass, Solver, Python_solver, Explain, Extract_topic, Combine_all, Combine_any
        if depth == max_depth:
            child_type = _pick_no_depth_increase_child(node_type)
        else:
            child_type = _pick_random_child(node_type)

        if child_type != "END":
            child = _rec(Node(child_type), depth, max_depth)
            node.add_child(child)
        return node

    # Start the recursive building process
    _rec(start, 0, max_depth)

    # Finally, add the END node to the last node
    end = Node("END")
    last_node = start
    while last_node.children:
        last_node = last_node.children[0]
    last_node.add_child(end)
    return start

def _generate_nx_graph(node, graph=None, parent=None, uid=None):
    if graph is None:
        graph = nx.DiGraph()
    
    # Assign a unique id if not given
    if uid is None:
        uid = id(node)  # use Python's unique object id
    
    # Add node with a label for visualization
    graph.add_node(uid, label=node.type_name)
    
    if parent:
        graph.add_edge(parent, uid)
    
    for child in node.children:
        _generate_nx_graph(child, graph, uid)
    
    return graph

def _hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    Positions nodes in a hierarchical layout for a DiGraph.
    Handles multiple children and ensures all nodes get a position.
    """
    if root is None:
        roots = [n for n, d in G.in_degree() if d == 0]
        if len(roots) == 0:
            raise ValueError("No root found")
        if len(roots) > 1:
            raise ValueError("Multiple roots found")
        root = roots[0]

    pos = {}
    visited = set()

    def _hierarchy_pos(node, width, vert_loc, xcenter):
        pos[node] = (xcenter, vert_loc)
        visited.add(node)
        neighbors = list(G.successors(node))
        if neighbors:
            dx = width / len(neighbors)
            nextx = xcenter - width/2 - dx/2
            for neighbor in neighbors:
                if neighbor not in visited:
                    nextx += dx
                    _hierarchy_pos(neighbor, dx, vert_loc - vert_gap, nextx)

    _hierarchy_pos(root, width, vert_loc, xcenter)

    # Print size of visited nodes for debugging
    print(f"Visited {len(visited)} nodes out of {len(G.nodes())}")
    return pos

def generate_random_graph(max_depth=2):
    graph = _build_random_graph(max_depth)
    nx_graph = _generate_nx_graph(graph)

    # Convert to Graph object
    nodes_with_types = [(node, data['label']) for node, data in nx_graph.nodes(data=True)]
    edges = list(nx_graph.edges())

    # Remap node ids to a contiguous range starting from 0
    id_mapping = {old_id: new_id for new_id, (old_id, _) in enumerate(nodes_with_types)}
    nodes_with_types = [(id_mapping[old_id], type_name) for old_id, type_name in nodes_with_types]
    edges = [(id_mapping[src], id_mapping[dst]) for src, dst in edges]

    graph_obj = Graph(nodes=nodes_with_types, edges=edges)
    return graph_obj


if __name__ == "__main__":
    g = generate_random_graph(max_depth=1)
    g.visualize()
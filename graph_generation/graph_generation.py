import sys
import os

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import logging
import random
from typing import Dict, Any, Tuple, Optional
from config.settings import Config
from data_management.graph_storage import Graph, GraphSet
from config.nodes import Node, NODE_TYPES, RULES, DEPTH_INCREASE_NODES, FIXED_COST
import networkx as nx
import time

def generate_graph_batch(
    config: Config,
    logger: logging.Logger,
    training_dataset: Optional[GraphSet] = None,
) -> Tuple[Dict[str, Any], GraphSet]:
    
    step_start = time.time()

    strategy = config.generation_strategy
    
    if strategy == 'random':
        graphset = _random_strategy(config, logger, training_dataset)
    # elif strategy == 'similar_to_training':
    #     graphset = _similar_to_training_strategy(config, logger, training_dataset)
    # elif strategy == 'custom':
    #     graphset = _custom_strategy(config, logger, training_dataset)
    else:
        raise ValueError(f"Unknown generation strategy: {strategy}")
    
    num_generated = graphset.size()

    duration = time.time() - step_start
    
    metrics = {
        'step_name': 'generation',
        'duration_seconds': round(duration, 4),
        'num_samples': num_generated,
        'strategy': strategy
    }
    
    return metrics, graphset

def _random_strategy(
    config: Config,
    logger: logging.Logger,
    training_dataset: Optional[GraphSet] = None,
) -> GraphSet:

    generated_graphs = GraphSet()
    
    for graph_idx in range(config.num_graphs_per_iteration):
        if (graph_idx + 1) % 1000 == 0:
            logger.debug(f"Generated {graph_idx + 1} / {config.num_graphs_per_iteration} graphs")
        try:
            graph = _random_graph(config.max_depth, config.max_nodes)
            if len(graph.get_nodes()) > config.max_nodes:
                # Graph exceeds max nodes, skip
                continue
            if generated_graphs.contains(graph):                            #
                # Duplicate graph, skip                                     #  These duplication checks are costly because it has to check for every element
                continue                                                    #  in the GraphSet, which uses list internally.
            if training_dataset and training_dataset.contains(graph):       #  TODO: check if this is a bottleneck and maybe change to set internally.
                # Duplicate of training data, skip                          #
                continue
            generated_graphs.add_graph(graph)
            
        except Exception as e:
            logger.error(f"Error generating graph {graph_idx}: {e}")
            continue
    
    return generated_graphs

# ---------------------------------------
# Random graph builder
# ---------------------------------------
def _build_random_graph(max_depth=2, max_nodes=20):
    start = Node("START")

    def _pick_random_child(node_type, remaining_nodes):
        allowed = RULES[node_type]["allowed_children"]
        a = []
        for child in allowed:
            cost = FIXED_COST.get(child, 1)
            if cost <= remaining_nodes:
                a.append( (child, cost) )
        if not a:
            # No child can be picked as there is no remaining nodes. Pick "END"
            return ("END", 0)
        return random.choice(a)
    
    def _pick_no_depth_increase_child(node_type, remaining_nodes):
        allowed = [child for child in RULES[node_type]["allowed_children"] if child not in DEPTH_INCREASE_NODES]
        a = []
        for child in allowed:
            cost = FIXED_COST.get(child, 1)
            if cost <= remaining_nodes:
                a.append( (child, cost) )
        if not a:
            # No child can be picked as there is no remaining nodes. Pick "END"
            return ("END", 0)
        return random.choice(a)

    def _rec(node, depth, max_depth, remaining_nodes):
        node_type = node.type_name
        rule = RULES[node_type]
        num_branches = rule["branches"]

        # For multi-branch nodes like Decompose_X or Split
        if node_type in ["Decompose_2", "Decompose_3", "Decompose_4", "Split"]:
            depth += 1
            children_to_combine = []
            remaining_nodes_per_branch = remaining_nodes // (num_branches + 1) + 1   # Evenly distribute remaining nodes across branches and Combine_all, +1 for the node in branches we already accounted for.
                
            for _ in range(num_branches):
                if depth == max_depth:
                    # Allow only nodes that dont lead to further depth increase
                    child_type, required_nodes = _pick_no_depth_increase_child(node_type, remaining_nodes_per_branch)
                else:
                    child_type, required_nodes = _pick_random_child(node_type, remaining_nodes_per_branch)   # these node_types for sure do not have "END" as allowed child, so no need for explicit check
                child = _rec(Node(child_type), depth, max_depth, remaining_nodes_per_branch - required_nodes)
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

            remaining_nodes_after_combine = remaining_nodes // (num_branches + 1)

            _rec(combine_all_node, depth, max_depth, remaining_nodes_after_combine)

            # Return the current node
            return node

        # For Validator
        if node_type == "Validator":
            # Two branches: True_pass and False_pass
            true_node = Node("True_pass")   # True branch just passes through
            false_node = _rec(Node("False_pass"), depth + 1, max_depth, remaining_nodes // 2) # False branch continues with recursive build
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

            _rec(combine_any_node, depth, max_depth, remaining_nodes // 2)

            # Return the current validator node
            return node
        
        # else, for single branch nodes (e.g. Solver, Python_solver, Explain, Extract_topic, etc.)
        if depth == max_depth:
            child_type, required_nodes = _pick_no_depth_increase_child(node_type, remaining_nodes)
        else:
            child_type, required_nodes = _pick_random_child(node_type, remaining_nodes)

        if child_type != "END":
            child = _rec(Node(child_type), depth, max_depth, remaining_nodes - required_nodes)
            node.add_child(child)
        return node

    # Start the recursive building process
    _rec(start, 0, max_depth, max_nodes)

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

def _random_graph(max_depth=2, max_nodes=20) -> Graph:
    graph = _build_random_graph(max_depth, max_nodes)
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
    generated_graphs = GraphSet()
    for i in range(10):
        g = _random_graph(max_depth=2, max_nodes=8)
        g.visualize()
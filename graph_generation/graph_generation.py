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
from config.nodes import Node, NODE_TYPES, RULES, DEPTH_INCREASE_NODES
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

    def _pick_random_child(node_type):
        allowed = RULES[node_type]["allowed_children"]
        return random.choice(allowed)
    
    def _pick_no_depth_increase_child(node_type):
        allowed = [child for child in RULES[node_type]["allowed_children"] if child not in DEPTH_INCREASE_NODES]
        return random.choice(allowed)

    def _rec(node, depth, max_depth, remaining_nodes):
        node_type = node.type_name
        rule = RULES[node_type]
        num_branches = rule["branches"]

        # For multi-branch nodes like Decompose_X or Split
        if node_type in ["Decompose_2", "Decompose_3", "Decompose_4", "Split"]:
            depth += 1
            children_to_combine = []
            for _ in range(num_branches):
                if remaining_nodes <= num_branches + 1:     # +1 for comining later on
                    child_type = _pick_no_depth_increase_child(node_type) # Pick random child that does not increase depth and stop with recursive build of the branch
                    child = Node(child_type)
                    node.add_child(child)
                    children_to_combine.append(child)
                else:
                    if depth == max_depth:
                        # Allow only nodes that dont lead to further depth increase
                        child_type = _pick_no_depth_increase_child(node_type)
                    else:
                        child_type = _pick_random_child(node_type)   # these node_types for sure do not have "END" as allowed child, so no need for explicit check
                    child = _rec(Node(child_type), depth, max_depth, remaining_nodes // (num_branches + 1) - 1)
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

            if remaining_nodes <= num_branches + 1:
                child_type = _pick_no_depth_increase_child("Combine_all")   # Pick random child that does not increase depth and stop with recursive build
                child = Node(child_type)
                combine_all_node.add_child(child)
            else:
                # Continue building from Combine_all
                _rec(combine_all_node, depth, max_depth, remaining_nodes // (num_branches + 1) - 1)

            # Return the current node
            return node

        # For Validator
        if node_type == "Validator":
            # Two branches: True_pass and False_pass
            true_node = Node("True_pass")   # True branch just passes through
            if remaining_nodes <= 2:    # 1 for false and 1 for combine_any
                false_child_type = _pick_no_depth_increase_child("False_pass") # Pick random child that does not increase depth and stop with recursive build of the branch
                false_node = Node("False_pass")
                false_node.add_child(Node(false_child_type))
            else:
                false_node = _rec(Node("False_pass"), depth + 1, max_depth, remaining_nodes // 2 - 1) # False branch continues with recursive build
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

            if remaining_nodes <= 2:
                child_type = _pick_no_depth_increase_child("Combine_any")
                if child_type != "END":
                    # Only add child if not END, as END will be added at the end
                    child = Node(child_type)
                    combine_any_node.add_child(child)
            else:
                # Continue building from Combine_any
                _rec(combine_any_node, depth, max_depth, remaining_nodes // 2 - 1)

            # Return the current validator node
            return node

        # For other nodes: False_pass, Solver, Python_solver, Explain, Extract_topic, Combine_all, Combine_any
        if remaining_nodes <= 1:
            child_type = _pick_no_depth_increase_child(node_type)
            if child_type != "END":
                child = Node(child_type)
                node.add_child(child)
                return node
            else:
                # If END is picked, dont add it here, will be added at the end. Just return current node
                return node
        
        if depth == max_depth:
            child_type = _pick_no_depth_increase_child(node_type)
        else:
            child_type = _pick_random_child(node_type)

        if child_type != "END":
            child = _rec(Node(child_type), depth, max_depth, remaining_nodes - 1)
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
    while True:
        g = _random_graph(max_depth=1, max_nodes=8)
        nodes = g.get_nodes()
        # Check if graph has no Solver or Python_solver nodes
        node_types = [node_type for _, node_type in nodes]
        if 'Solver' not in node_types and 'Python_solver' not in node_types:
            g.visualize()
            break

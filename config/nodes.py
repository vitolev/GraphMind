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
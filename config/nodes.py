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
    "False_pass": {"branches": 1, "allowed_children": ["Solver", "Python_solver", "Explain", "Extract_topic", "Decompose_2", "Decompose_3", "Decompose_4"]},
    
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

EDGES = [
    ("Combine_all","Extract_topic"),
    ("Split","Decompose_3"),
    ("Combine_all","Combine_all"),
    ("Decompose_2","Explain"),
    ("Decompose_2","Solver"),
    ("Combine_all","Decompose_3"),
    ("Extract_topic","END"),
    ("Extract_topic","Split"),
    ("Combine_any","Python_solver"),
    ("Validator","False_pass"),
    ("False_pass","Extract_topic"),
    ("Combine_any","Decompose_2"),
    ("START","Decompose_2"),
    ("START","Decompose_4"),
    ("Explain","Python_solver"),
    ("False_pass","Decompose_3"),
    ("Decompose_3","Python_solver"),
    ("Extract_topic","Combine_any"),
    ("Decompose_4","Decompose_2"),
    ("False_pass","Solver"),
    ("Decompose_3","Decompose_2"),
    ("Extract_topic","Python_solver"),
    ("START","Split"),
    ("Combine_any","Extract_topic"),
    ("Explain","END"),
    ("Decompose_4","Split"),
    ("Split","Explain"),
    ("Explain","Split"),
    ("Decompose_3","Split"),
    ("Split","Solver"),
    ("Combine_any","Combine_any"),
    ("Decompose_2","Decompose_2"),
    ("Decompose_2","Decompose_4"),
    ("Decompose_3","Extract_topic"),
    ("Combine_all","Explain"),
    ("Combine_all","Solver"),
    ("START","Python_solver"),
    ("Extract_topic","Combine_all"),
    ("Explain","Combine_any"),
    ("Decompose_4","Python_solver"),
    ("Decompose_2","Split"),
    ("False_pass","Explain"),
    ("Solver","END"),
    ("Solver","Validator"),
    ("True_pass","Combine_any"),
    ("START","Extract_topic"),
    ("Combine_any","Combine_all"),
    ("Combine_any","Decompose_3"),
    ("Decompose_2","Python_solver"),
    ("Decompose_4","Extract_topic"),
    ("START","Decompose_3"),
    ("Combine_any","Explain"),
    ("Explain","Extract_topic"),
    ("Combine_any","Solver"),
    ("Split","Decompose_4"),
    ("Solver","Combine_any"),
    ("Explain","Combine_all"),
    ("Validator","True_pass"),
    ("Decompose_4","Decompose_3"),
    ("Python_solver","END"),
    ("Combine_all","Decompose_4"),
    ("Combine_all","Decompose_2"),
    ("Decompose_4","Solver"),
    ("Decompose_3","Decompose_3"),
    ("Decompose_3","Explain"),
    ("Explain","Solver"),
    ("Decompose_3","Solver"),
    ("Combine_all","END"),
    ("Extract_topic","Explain"),
    ("Decompose_2","Extract_topic"),
    ("Extract_topic","Solver"),
    ("False_pass","Decompose_2"),
    ("False_pass","Decompose_4"),
    ("Python_solver","Combine_any"),
    ("Decompose_2","Decompose_3"),
    ("Solver","Combine_all"),
    ("Split","Python_solver"),
    ("Combine_all","Combine_any"),
    ("START","Explain"),
    ("Split","Decompose_2"),
    ("Combine_all","Python_solver"),
    ("START","Solver"),
    ("Combine_any","Decompose_4"),
    ("Decompose_4","Explain"),
    ("Python_solver","Validator"),
    ("Combine_any","END"),
    ("False_pass","Combine_any"),
    ("Decompose_3","Decompose_4"),
    ("Python_solver","Combine_all"),
    ("Split","Extract_topic"),
    ("False_pass","Python_solver")
]

FIXED_COST = {
    "Decompose_2": 4,   # 1 Decompose_2, 1 Combine_all, 2 branches
    "Decompose_3": 5,   # 1 Decompose_3, 1 Combine_all, 3 branches
    "Decompose_4": 6,   # 1 Decompose_4, 1 Combine_all, 4 branches
    "Split":       4,   # 1 Split, 1 Combine_all, 2 branches
    "Validator":   4,   # 1 Validator, 1 Combine_any, 1 True_pass, 1 False_pass
    "END":         0    # END node has no cost
}

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
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
    ("START", "Solver"),
    ("Explain", "Solver"),
    ("Decompose_4", "Decompose_2"),
    ("Combine_any", "Decompose_3"),
    ("Combine_all", "Decompose_3"),
    ("Decompose_4", "Decompose_4"),
    ("Python_solver", "Validator"),
    ("Split", "Solver"),
    ("Combine_any", "Python_solver"),
    ("Extract_topic", "Python_solver"),
    ("Decompose_2", "Python_solver"),
    ("Decompose_3", "Solver"),
    ("Combine_any","Combine_all"),
    ("True_pass","Combine_any"),
    ("START","Decompose_2"),
    ("START","Decompose_4"),
    ("Python_solver","Combine_all"),
    ("Decompose_4","Decompose_3"),
    ("Decompose_2","Decompose_4"),
    ("Combine_any","Explain"),
    ("Split","Decompose_2"),
    ("Combine_any","Extract_topic"),
    ("Decompose_4","Python_solver"),
    ("Combine_all","Extract_topic"),
    ("Extract_topic","Explain"),
    ("Decompose_2","Explain"),
    ("Split","Decompose_4"),
    ("Decompose_3","Decompose_2"),
    ("Solver","END"),
    ("Decompose_3","Decompose_4"),
    ("Solver","Combine_any"),
    ("START","Decompose_3"),
    ("Extract_topic","Split"),
    ("START","Python_solver"),
    ("Explain","Python_solver"),
    ("Decompose_2","Decompose_3"),
    ("Solver","Validator"),
    ("Decompose_4","Explain"),
    ("Split","Decompose_3"),
    ("END","Combine_all"),
    ("Split","Python_solver"),
    ("Decompose_3","Decompose_3"),
    ("False_pass","Solver"),
    ("Validator","False_pass"),
    ("Decompose_3","Python_solver"),
    ("Solver","Combine_all"),
    ("Decompose_4","Split"),
    ("START","Explain"),
    ("START","Extract_topic"),
    ("Combine_all","Solver"),
    ("Decompose_2","Extract_topic"),
    ("Split","Explain"),
    ("False_pass","Decompose_2"),
    ("START","Split"),
    ("Decompose_3","Explain"),
    ("Explain","Split"),
    ("False_pass","Decompose_4"),
    ("Decompose_2","Split"),
    ("Decompose_4","Extract_topic"),
    ("Combine_all","Decompose_2"),
    ("False_pass","Decompose_3"),
    ("False_pass","Python_solver"),
    ("Explain","Extract_topic"),
    ("Combine_any","Solver"),
    ("Extract_topic","Solver"),
    ("Decompose_2","Solver"),
    ("Split","Extract_topic"),
    ("Combine_all","Python_solver"),
    ("Decompose_3","Extract_topic"),
    ("False_pass","Explain"),
    ("False_pass","Extract_topic"),
    ("Combine_any","Decompose_2"),
    ("Decompose_4","Solver"),
    ("Decompose_3","Split"),
    ("Combine_any","Decompose_4"),
    ("Combine_all","Decompose_4"),
    ("Decompose_2","Decompose_2"),
    ("Combine_all","Explain"),
    ("Combine_any","END"),
    ("Validator","True_pass"),
    ("Combine_any","Combine_any"),
    ("END","Combine_any"),
    ("Python_solver","END"),
    ("Python_solver","Combine_any")
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
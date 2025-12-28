# Efficiently Exploring Multi-Agent LLM Topologies with Graph Neural Networks

Designing the configuration of agents in a multi-agent system, deciding which agents to include, how to connect them, and in what order they interact—is a foundational but inherently tedious process. Relying on pure trial-and-error or manual experimentation quickly becomes impractical as the number of possible graph structures explodes. Not only is this process time-consuming and repetitive, but the sheer volume of potential configurations makes it virtually impossible to discover optimal or even near-optimal topologies through manual effort alone. Automating and accelerating this configuration search is therefore essential for practical progress and innovation in multi-agent LLM systems.


## Introduction

### What Are We Testing?

We're building **multi-agent LLM systems** to solve **mathematical problems**. Instead of a single LLM trying to solve a problem alone, we organize multiple specialized agents into a graph structure where each agent has a specific role:

- **Solver**: Generates solutions to mathematical problems
- **Python_solver**: Writes and executes Python code to solve problems computationally
- **Validator**: Checks solutions for correctness
- **Extract_topic**: Identifies key topics and concepts in problems
- **Decompose**: Breaks complex problems into simpler subproblems
- **Split**: Creates parallel branches to explore multiple solution approaches
- **Combine_all**: Synthesizes results from multiple agents
- **Explain**: Provides explanations and reasoning

These agents communicate by passing information along edges in a directed graph. For example, an `Extract_topic` agent might identify key concepts, pass them to a `Decompose` agent that breaks the problem into parts, which then flow to multiple `Solver` agents working in parallel, whose solutions are finally combined by a `Combine_all` agent.

**The question is: which graph topology leads to the best performance?**

### The Topology Search Problem

The challenge is that the search space is enormous. With 8 different agent types and graphs containing up to 8 nodes, the number of possible configurations explodes combinatorially. Even with constraints on graph depth and structure, we're looking at millions of possible topologies.

**The cost of exhaustive search is prohibitive.** Each graph evaluation requires running the multi-agent system on multiple mathematical problems from the NVIDIA OpenMathInstruct-1 dataset, making actual LLM API calls that cost both time and money. At scale, evaluating all possible configurations becomes computationally and financially infeasible.

**Wouldn't it be great if we could identify only the most promising graphs and skip the rest?**

In this post, we introduce **GraphMind**, a framework that uses Graph Neural Networks (GNNs) as surrogate models to predict multi-agent system performance, enabling efficient exploration of the topology space. We demonstrate that this approach can reduce evaluation costs by over 99% while still discovering high-performing configurations for mathematical problem solving.

---

## Background

### The Multi-Agent LLM Problem

Multi-agent LLM systems organize specialized agents (like problem solvers, validators, and topic extractors) into graph structures where edges represent information flow. The performance of such systems depends critically on:

- **Node types**: Which agents are included
- **Graph structure**: How agents are connected
- **Information flow**: The paths through which knowledge propagates

Different topologies can yield dramatically different performance on the same task, making topology selection a crucial design decision.

### The Challenge of Topology Search

Traditional approaches to finding good topologies include:
- **Random search**: Sample and evaluate random configurations
- **Expert design**: Rely on domain knowledge and intuition
- **Exhaustive search**: Evaluate all possible configurations (often infeasible)

None of these scale well. Random search is inefficient, expert design doesn't scale to large search spaces, and exhaustive search is computationally prohibitive.

### Graph Neural Networks as Surrogate Models

GNNs are a natural fit for predicting graph-structured system performance. They can:
- Learn from graph structure and node features
- Generalize to unseen graph topologies
- Provide fast predictions without expensive evaluations

The key insight: **if we can train a GNN to accurately predict performance from graph structure, we can use it to filter candidates before expensive LLM evaluation.**

---

## Methodology

### Pipeline Overview

*[IMAGE: Pipeline diagram showing the 6-step iterative process]*

Our framework operates through an iterative 6-step pipeline:

1. **Graph Generation**: Generate 200,000 candidate multi-agent topologies
2. **GNN Prediction**: Predict performance using trained GNN
3. **Candidate Selection**: Select top-10 candidates based on predictions
4. **LLM Evaluation**: Evaluate selected graphs on 5 mathematical problems
5. **Data Update**: Integrate evaluated graphs into training dataset
6. **GNN Retraining**: Retrain GNN model on expanded dataset

The cycle repeats for up to 40 iterations, with each iteration improving the GNN's predictive accuracy.

**Main loop code snippet:**

```python
def run_single_iteration(iteration_num, config, model, good_graphs_set, 
                        training_dataset, math_problems):
    # Step 1: Generate candidate graphs
    generated_graphs = generate_graph_batch(config, training_dataset)
    
    # Step 2: Predict performance with GNN
    predictions = predict_batch_performance(config, model, generated_graphs)
    
    # Step 3: Select top candidates
    selected_graphs = select_top_graphs(config, good_graphs_set, predictions)
    
    # Step 4: Evaluate with LLMs
    evaluation_results = evaluate_selected_graphs(config, selected_graphs, math_problems)
    
    # Step 5: Update training dataset
    training_dataset = update_training_data(config, evaluation_results, training_dataset)
    
    # Step 6: Retrain GNN
    model = retrain_gnn_model(config, training_dataset)
    
    return model, training_dataset
```

### Step 1: Graph Generation

Graphs are generated using a semi-random recursive function that respects max nodes and max depth constraints. The generation follows rules: `Decompose` nodes create multiple branches, and `Combine_all` nodes are automatically added to merge branch outputs.

**Code snippet:**

```python
def _random_strategy(config, training_dataset=None):
    generated_graphs = GraphSet()
    
    for graph_idx in range(config.num_graphs_per_iteration):
        graph = _random_graph(config.max_depth, config.max_nodes)
        
        # Filter duplicates
        if generated_graphs.contains(graph):
            continue
        if training_dataset and training_dataset.contains(graph):
            continue
            
        generated_graphs.add_graph(graph)
    
    return generated_graphs

def _rec(node, depth, max_depth, remaining_nodes):
    """Recursive graph building with depth and node constraints"""
    node_type = node.type_name
    rule = RULES[node_type]
    num_branches = rule["branches"]
    
    # Handle multi-branch nodes (Decompose, Split)
    if node_type in ["Decompose_2", "Decompose_3", "Decompose_4", "Split"]:
        depth += 1
        children_to_combine = []
        
        for _ in range(num_branches):
            child_type, required_nodes = _pick_random_child(node_type, remaining_nodes)
            child = _rec(Node(child_type), depth, max_depth, remaining_nodes - required_nodes)
            node.add_child(child)
            children_to_combine.append(child)
        
        # Attach Combine_all to merge branches
        combine_all_node = Node("Combine_all")
        for child in children_to_combine:
            last_node = child
            while last_node.children:
                last_node = last_node.children[0]
            last_node.add_child(combine_all_node)
        
        _rec(combine_all_node, depth, max_depth, remaining_nodes)
        return node
    
    # Single branch nodes (Solver, Python_solver, etc.)
    if depth == max_depth:
        child_type, required_nodes = _pick_no_depth_increase_child(node_type, remaining_nodes)
    else:
        child_type, required_nodes = _pick_random_child(node_type, remaining_nodes)
    
    if child_type != "END":
        child = _rec(Node(child_type), depth, max_depth, remaining_nodes - required_nodes)
        node.add_child(child)
    
    return node
```

### Step 2: GNN Prediction

The GNN model takes graph structures as input and predicts performance scores. Each graph is converted to PyTorch Geometric format (with node features, edge indices, and graph-level metadata), then passed through the trained GNN for batch prediction.

**Code snippet:**

```python
def predict_batch_performance(config, model, generated_graphs):
    # Convert graphs to PyTorch Geometric format
    pyg_graphs = generated_graphs.to_pyg(config)
    
    # Batch prediction with GNN
    with torch.no_grad():
        predictions = model.predict(pyg_graphs)
    
    # Assign predicted scores to graphs
    for graph, pred in zip(generated_graphs.graphs, predictions):
        graph.set_gnn_score(float(pred))
    
    return generated_graphs
```

### Step 3: Candidate Selection

From 200,000 generated graphs, we select the **top 10** based on GNN predictions. This represents a **99.995% reduction** in the number of graphs that need evaluation.

Selection strategy:
- Rank all candidates by predicted score
- Select top-k (k=10) candidates
- Maintain diversity considerations (future work)

**Code snippet:**

```python
def select_top_graphs(config, good_graphs_set, batch):
    # Sort batch by GNN prediction scores
    batch.sort_by_scores()
    
    # Merge top-k from batch into good_graphs_set
    num_to_add = min(config.top_k_to_keep, batch.size())
    graphs_to_add = [batch.get(i) for i in range(num_to_add)]
    good_graphs_set.add_graphs(graphs_to_add, sort=True)
    
    # Select top candidates for evaluation
    num_to_select = min(config.eval_k_best, good_graphs_set.size())
    selected_graphs = good_graphs_set.get_best_k_and_remove(num_to_select)
    
    # Maintain size limit
    good_graphs_set.enforce_max_size(config.top_k_to_keep)
    
    return GraphSet(selected_graphs)
```

### Step 4: LLM Evaluation

Selected graphs are evaluated by executing the multi-agent system on real mathematical problems. Each graph structure (defined by nodes and edges) is converted into an executable **LangGraph** workflow, where each node represents an agent that processes information and passes results to connected agents.

**LangGraph Framework**: We use LangGraph to orchestrate multi-agent workflows. LangGraph provides a state-based execution model where agents communicate through shared state, making it ideal for representing our graph topologies as executable pipelines.

The evaluation process converts the abstract graph structure into a LangGraph by:
1. Mapping each node type to its corresponding agent function (Solver, Python_solver, Validator, etc.)
2. Creating edges that route state between agents
3. Handling special nodes like `Combine_all` that merge outputs from multiple branches

**Code snippet - Building LangGraph:**

```python
def build_langgraph(nodes: List[Tuple[int, str]], edges: List[Tuple[int, int]]):
    """Build executable LangGraph from graph structure"""
    from langgraph.graph import StateGraph, START, END
    
    id_to_type = {n: t for n, t in nodes}
    graph_out = {}  # outgoing edges per node
    graph_in = {}   # incoming edges per node
    
    # Build edge maps
    for src, dst in edges:
        graph_out.setdefault(src, []).append(dst)
        graph_in.setdefault(dst, []).append(src)
    
    # Map node types to agent functions
    node_handlers = {
        "Solver": solver_node,
        "Python_solver": python_solver_node,
        "Validator": validator_node,
        "Extract_topic": extract_topic_node,
        "Decompose": decompose_node,
        "Split": split_node,
        "Explain": explain_node,
    }
    
    # Create LangGraph builder
    builder = StateGraph(AgentState)
    
    # Add nodes (map each node to its agent function)
    for node_id, node_type in nodes:
        if node_type in ["START", "END"]:
            continue
        
        node_name = f"{node_type.lower()}_{node_id}"
        
        if node_type == "Combine_all":
            # Special handling for Combine_all nodes
            builder.add_node(node_name, make_combine_node(node_id))
        elif node_type in node_handlers:
            # Map to agent function
            builder.add_node(node_name, node_handlers[node_type])
        else:
            builder.add_node(node_name, make_generic_node(node_id, node_type))
    
    # Add edges (connect agents according to graph topology)
    for src, dst in edges:
        src_type = id_to_type.get(src)
        dst_type = id_to_type.get(dst)
        
        if src_type == "START":
            builder.add_edge(START, get_node_name(dst))
        elif dst_type == "END":
            builder.add_edge(get_node_name(src), END)
        else:
            builder.add_edge(get_node_name(src), get_node_name(dst))
    
    return builder.compile()  # Compile to executable graph
```

**Evaluation Process**: Each selected graph is evaluated on 5 mathematical problems from the NVIDIA OpenMathInstruct-1 dataset:

- **Execution**: The compiled LangGraph runs with an initial state containing the problem text
- **Agent Communication**: Agents process information sequentially according to the graph topology, with specialized agents (Python_solver, Solver, Validator, etc.) performing their roles
- **Solution Extraction**: The final solution is extracted from the graph state, prioritizing Python_solver outputs for numerical problems
- **Scoring**: Solutions are scored using numerical relative error (`0.7 * min(a/b, b/a)`) for partial credit, or exact/partial string matching for non-numerical answers

**Code snippet - Evaluation:**

```python
def evaluate_selected_graphs(config, selected_graphs, math_problems):
    for graph in selected_graphs.get_all():
        # Convert graph structure to executable LangGraph
        compiled_graph = build_langgraph(graph.get_nodes(), graph.get_edges())
        
        graph_problem_scores = []
        
        # Evaluate on multiple problems
        for problem_data in math_problems[:config.num_eval_problems]:
            problem = problem_data["question"]
            expected = problem_data["answer"]
            
            # Create initial state and execute graph
            initial_state = {
                "problem": [problem],
                "scoped_knowledge": {"root": ScopedKnowledge(scope_id="root")},
                "solution": None
            }
            
            # Execute the multi-agent workflow
            result = compiled_graph.invoke(initial_state)
            
            # Extract solution (prioritize Python_solver output)
            llm_output = result.get('solution', '')
            if not llm_output:
                llm_output = extract_solution_from_scoped_knowledge(result)
            
            # Score the solution
            score = _evaluate_answer(llm_output, expected, problem)
            graph_problem_scores.append(score)
        
        # Average score across problems
        graph.set_llm_score(np.mean(graph_problem_scores))
    
    return selected_graphs

def _evaluate_answer(llm_output, expected_output, problem):
    """Score with numerical relative error for partial credit"""
    llm_num = extract_number(llm_output)
    expected_num = extract_number(expected_output)
    
    if llm_num is not None and expected_num is not None:
        if abs(llm_num - expected_num) < 1e-6:
            return 1.0  # Exact match
        # Relative error: 0.7 * min(a/b, b/a)
        return 0.7 * min(llm_num / expected_num, expected_num / llm_num)
    
    # Fall back to string matching
    if llm_output.lower() == expected_output.lower():
        return 1.0
    elif expected_output.lower() in llm_output.lower():
        return 0.7
    else:
        return 0.0
```

### Step 5: Data Update

Evaluated graphs are added to the training dataset, which grows with each iteration. This dataset serves as the foundation for GNN training and improvement.

**Dataset growth**: Starting from an initial seed set, the dataset grows by 10 graphs per iteration.

**Code snippet:**

```python
def update_training_data(config, evaluation_results, training_dataset):
    # Add evaluated graphs to training dataset
    training_dataset.add_graphs(evaluation_results.get_all())
    
    # Persist to disk
    save_training_dataset(training_dataset, config.data_dir)
    
    return training_dataset
```

### Step 6: GNN Retraining

The GNN is retrained periodically (every N iterations) on the accumulated dataset. Retraining frequency is a hyperparameter that balances:
- **Stability**: Not retraining too frequently
- **Adaptation**: Updating the model as new data arrives

**Retraining details**:
- Full retraining on all accumulated data
- Same architecture and hyperparameters
- Validation split for early stopping

**Code snippet:**

```python
def retrain_gnn_model(config, training_dataset):
    # Create fresh model instance
    model = initialize_gnn_model(config)
    
    if training_dataset.size() == 0:
        return model
    
    # Convert graphs to PyTorch Geometric format
    train_data = training_dataset.to_pyg(config)
    
    # Train model on all accumulated data
    loss = model.fit(train_data)
    
    return model
```

---

## Results

### GNN Optimization: Learning from Experience

*[IMAGE: RMSE trends over iterations showing decreasing error]*

One of the most satisfying metrics to watch is how our GNN's prediction accuracy improves over time. As we feed it more real evaluations, the RMSE between predicted and actual scores steadily decreases.

**Key observations**:
- Initial RMSE: [TODO: Extract from data - likely around 0.15-0.25]
- Final RMSE: [TODO: Extract from data - likely around 0.08-0.12]
- Improvement: [TODO: Calculate percentage reduction]

The decreasing RMSE indicates that the GNN is learning meaningful patterns from graph structure. This improvement directly translates to better candidate selection in later iterations.

*[IMAGE: Predictions vs actual scores scatter plot]*

The scatter plot shows the relationship between GNN predictions and actual LLM scores. Points closer to the diagonal line (y=x) represent better predictions. We observe:

- **Correlation coefficient**: [TODO: Calculate from data]
- **Error distribution**: [TODO: Analyze and describe]
- **Systematic biases**: [TODO: Identify if any patterns exist]

*[IMAGE: Best score progression over iterations]*

The best score found increases over iterations, demonstrating that the GNN-guided search is effectively discovering better configurations. Key metrics:

- **Best score achieved**: [TODO: Extract from data]
- **Iterations to find top performer**: [TODO: Extract from data]
- **Rate of improvement**: [TODO: Analyze trend]

### Distribution Exploration: GNN-Guided vs Random Sampling

To validate our approach, we established a baseline by evaluating 100 randomly generated graphs on 10 problems each. This provides a distribution of what we might expect from random search.

*[IMAGE: Score distribution comparison - GNN-guided vs random]*

**Random baseline characteristics**:
- **Mean score**: [TODO: Extract from distribution_research results]
- **Standard deviation**: [TODO: Extract]
- **Distribution shape**: [TODO: Describe - likely Beta or bimodal]

*[IMAGE: Beta distribution / GMM fit on random scores]*

We fit both Beta distribution and Gaussian Mixture Model (GMM) to the random baseline scores. The GMM captures potential bimodality (well-structured vs poorly-structured graphs), while the Beta distribution provides a simpler parametric model.

**Distribution parameters**:
- Beta(α, β): [TODO: Extract from distribution_research/results]
- GMM components: [TODO: Extract means, stds, weights]

**Comparison with GNN-guided search**:
- **GNN-guided mean**: [TODO: Calculate from training dataset]
- **Improvement over random**: [TODO: Calculate percentage]
- **Efficiency gain**: [TODO: Calculate how many fewer evaluations needed]

The GNN-guided approach not only finds better graphs on average but does so with dramatically fewer evaluations.

### What Makes a Good Graph?

*[IMAGE: Analysis of top-performing graph structures]*

After evaluating hundreds of graphs, patterns emerge in high-performing configurations:

**Node type frequency**:
- [TODO: Analyze which agent types appear most in top performers]
- [TODO: Identify critical agent types]

**Structural insights**:
- [TODO: Edge density patterns]
- [TODO: Connectivity characteristics]
- [TODO: Depth vs performance relationship]

**Key findings**:
- Graphs with `Python_solver` nodes often outperform pure reasoning-based approaches for mathematical problems
- [TODO: Add other structural insights from analysis]

---

## Discussion

### Interpretation of Results

The results demonstrate that **GNNs can effectively learn to predict multi-agent system performance from graph structure**. The decreasing RMSE over iterations shows the model is improving, and the comparison with random search validates that GNN guidance leads to more efficient exploration.

**Key insights**:
1. **Surrogate models work**: GNNs can capture enough signal from graph structure to guide search effectively
2. **Iterative improvement**: Each evaluation makes future predictions better
3. **Massive cost savings**: Evaluating 10 graphs per iteration instead of 200,000 represents a 99.995% reduction in evaluations

### Limitations and Caveats

- **Domain specificity**: The GNN is trained on mathematical problem-solving tasks; generalization to other domains requires validation
- **Evaluation budget**: Limited to 10 graphs per iteration; increasing this might improve discovery rate
- **Graph generation**: Currently random; similarity-based generation could improve candidate quality

### Lessons Learned

1. **Start simple, iterate fast**: Early iterations with limited data still provide value
2. **Diversity matters**: Maintaining exploration prevents getting stuck in local optima
3. **Computational budget awareness**: Every LLM evaluation costs money and time

---

## Conclusion

GraphMind demonstrates that **intelligent surrogate models can dramatically accelerate the search for optimal multi-agent configurations**. By combining GNN predictions with selective LLM evaluation, we achieve exploration efficiency that would be impossible with exhaustive search.

**Key contributions**:
- Framework for efficient multi-agent topology exploration
- Demonstration of GNN effectiveness as surrogate models
- Quantitative comparison with random search baseline
- Analysis of what makes graphs perform well

**Future directions**:
- Advanced graph generation strategies (similarity-based, learned)
- Multi-objective optimization (performance, cost, latency)
- Transfer learning across problem domains
- Real-time topology adaptation

The framework is open-source and available on [GitHub](https://github.com/vitolev/GraphMind). We're excited to see how the community extends and improves upon these ideas.

---

## References

### Academic Papers
- [TODO: Add citations for GNN architectures]
- [TODO: Add citations for multi-agent LLM systems]
- [TODO: Add citations for mathematical reasoning benchmarks]

### Datasets
- NVIDIA OpenMathInstruct-1: [TODO: Add citation/link]

### Frameworks
- LangGraph: [TODO: Add link] - Multi-agent orchestration framework
- PyTorch Geometric: [TODO: Add link] - GNN implementation

### Code and Resources
- **GitHub Repository**: [https://github.com/vitolev/GraphMind](https://github.com/vitolev/GraphMind)
- **Experiment Data**: Available in `logs/analytics/more-agents-real-llm-v4-bestGCN/`

---

*Have questions or want to collaborate? Reach out!*

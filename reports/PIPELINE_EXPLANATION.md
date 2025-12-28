# Comprehensive Explanation of the GraphMind Pipeline

This document provides a thorough, detailed explanation of the 6-step iterative pipeline used in the GraphMind framework for efficiently exploring multi-agent LLM system topologies.

---

## Overview

The GraphMind pipeline is an iterative optimization system that uses Graph Neural Networks (GNNs) as surrogate models to efficiently explore the space of multi-agent LLM graph topologies. The pipeline operates through 6 sequential steps that repeat for a configurable number of iterations (typically 40 iterations), progressively improving the GNN's ability to predict which graph structures will perform well on mathematical problem-solving tasks.

**Core Concept**: Instead of evaluating all possible graph configurations with expensive LLM calls, the system uses a GNN to predict performance, evaluates only the top candidates, and uses those results to improve future predictions.

---

## Data Structures and State

Before diving into the steps, it's crucial to understand the key data structures that persist across iterations:

### Graph Object
Each graph is represented as a `Graph` object containing:
- **nodes**: List of (node_id, node_type) tuples representing agents in the multi-agent system
- **edges**: List of (from_node_id, to_node_id) tuples representing directed information flow
- **gnn_score**: Float (0.0 to 1.0) - Predicted performance score from GNN
- **llm_score**: Float (0.0 to 1.0) - Actual performance score from LLM evaluation (set to 0.0 if not yet evaluated)
- **time_evaluating**: Float - Time taken to evaluate this graph with LLMs

### GraphSet
A container class that holds multiple `Graph` objects:
- Maintains graphs in a list
- Can sort graphs by scores (llm_score first, then gnn_score, descending)
- Provides methods to add/remove graphs, check for duplicates, and convert to PyTorch Geometric format

### Key Persistent State

1. **training_dataset** (GraphSet): 
   - Contains all graphs that have been evaluated with LLMs
   - Grows with each iteration as new evaluations are added
   - Used to train/retrain the GNN model
   - Persisted to disk as `data/training_dataset.pkl`

2. **good_graphs_set** (GraphSet):
   - Contains the top-k graphs (by GNN prediction scores) from recent iterations
   - Acts as a temporary buffer of promising candidates
   - Limited to `top_k_to_keep` graphs (e.g., 15 graphs)
   - Persisted to disk as `data/good_graphs_set.pkl`

3. **model** (GNN Model Object):
   - The trained GNN model (GCN, GAT, HetGAT, or GraphSAGE)
   - Used for predictions in Step 2
   - Retrained in Step 6 on the updated training_dataset

---

## Pipeline Initialization

Before iterations begin, the pipeline performs initialization:

1. **Load Configuration**: Reads parameters from `config/experiment_config.yaml` including:
   - `num_graphs_per_iteration`: Number of graphs to generate per iteration (e.g., 200,000)
   - `max_nodes`: Maximum nodes per graph (e.g., 8)
   - `max_depth`: Maximum graph depth (e.g., 3)
   - `top_k_to_keep`: Number of top graphs to keep in good_graphs_set (e.g., 15)
   - `eval_k_best`: Number of graphs to evaluate with LLMs per iteration (e.g., 10)
   - `num_eval_problems`: Number of math problems to evaluate each graph on (e.g., 5)
   - `max_iterations`: Total number of iterations to run (e.g., 40)

2. **Initialize GNN Model**: Creates a new GNN model instance based on `gnn_model_type` config

3. **Load Training Dataset**: Attempts to load existing `training_dataset.pkl` if it exists. If present and non-empty, the GNN is retrained on this data before starting iterations.

4. **Load Math Problems**: Loads mathematical problems from NVIDIA OpenMathInstruct-1 dataset (subset based on `num_eval_problems` and `seed`)

5. **Load Good Graphs Set**: Attempts to load existing `good_graphs_set.pkl` if it exists (may be empty initially)

6. **Initialize Pipeline State**: Creates a `PipelineState` object to track:
   - Current iteration number
   - Total graphs generated across all iterations
   - Total evaluations performed
   - Training dataset size
   - Iteration history (list of metrics dictionaries)

---

## The 6-Step Iterative Pipeline

Each iteration follows the same 6-step sequence. Here's a detailed breakdown of each step:

---

### STEP 1: Graph Generation

**Purpose**: Generate a large batch of candidate graph topologies to explore.

**Function**: `generate_graph_batch(config, logger, training_dataset)`

**Process**:
1. **Generation Strategy**: Currently uses "random" strategy (other strategies like "similar_to_training" may be implemented in the future)

2. **Random Graph Construction**:
   - Uses a recursive tree-building algorithm starting from a "START" node
   - At each node, randomly selects a child node type based on:
     - Valid child types according to graph rules (defined in `config/nodes.py`)
     - Remaining node budget (ensures graphs don't exceed `max_nodes`)
     - Depth constraints (ensures graphs don't exceed `max_depth`)
   - Special handling for nodes that create multiple branches (e.g., `Split`, `Decompose`, `Validator` with True/False branches)
   - Eventually terminates at "END" node
   - Converts the tree structure to a directed graph representation

3. **Duplicate Filtering**:
   - For each generated graph, checks if it's a duplicate of:
     - Other graphs in the current batch (using `generated_graphs.contains()`)
     - Graphs already in the training dataset (to avoid re-evaluating known graphs)
   - Skips duplicates and continues generating until reaching `num_graphs_per_iteration` unique graphs

4. **Output**:
   - Returns a `GraphSet` containing all generated graphs (typically 200,000 graphs)
   - All graphs have `llm_score = 0.0` (not yet evaluated)
   - All graphs have `gnn_score = 0.0` (not yet predicted)

**Metrics Collected**:
- Duration in seconds
- Number of graphs generated
- Generation strategy used

**Example**: Generates 200,000 unique graph structures like:
- Graph A: START → Extract_topic → Solver → END
- Graph B: START → Python_solver → Validator → Solver → Combine_all → END
- Graph C: START → Decompose → [Solver, Solver] → Combine_all → END
- ... (199,997 more unique graphs)

---

### STEP 2: GNN Prediction

**Purpose**: Use the trained GNN model to predict performance scores for all generated graphs without expensive LLM evaluation.

**Function**: `predict_batch_performance(config, logger, model, generated_graphs)`

**Process**:
1. **Graph Format Conversion**:
   - Converts each `Graph` object to PyTorch Geometric format (`Data` or `HeteroData` depending on `config.data_format`)
   - For heterogeneous graphs (HetGAT), creates separate node features for each node type
   - For homogeneous graphs (GCN/GAT), uses uniform node features
   - Includes graph structure information (edges, node types) and metadata (number of nodes, edges, depth)

2. **Batch Prediction**:
   - Passes all graphs through the GNN model in a single batch (or multiple batches if too large)
   - Model performs forward pass:
     - For GCN/GAT: Applies graph convolution/attention layers, aggregates to graph-level representation, outputs scalar prediction
     - For HetGAT: Handles different node types separately, applies type-specific transformations, aggregates, outputs prediction
   - Predictions are floats in range [0.0, 1.0] representing expected performance

3. **Score Assignment**:
   - For each graph, stores the predicted score in `graph.gnn_score`
   - The original `GraphSet` is modified in-place (graphs now have predictions)

4. **Output**:
   - Returns the same `GraphSet` with all graphs now having `gnn_score` values set
   - Graphs are NOT sorted by score yet (sorting happens in Step 3)

**Metrics Collected**:
- Duration (inference time)
- Number of graphs predicted
- Best predicted score (maximum `gnn_score`)
- Worst predicted score (minimum `gnn_score`)
- Mean predicted score
- Standard deviation of predictions
- Inference time per graph

**Example**: 
- Graph A gets `gnn_score = 0.45` (model predicts mediocre performance)
- Graph B gets `gnn_score = 0.82` (model predicts good performance)
- Graph C gets `gnn_score = 0.91` (model predicts excellent performance)
- ... (predictions for all 200,000 graphs)

**Key Point**: This step is very fast (seconds) compared to LLM evaluation (minutes/hours), which is why it's feasible to predict scores for hundreds of thousands of graphs.

---

### STEP 3: Candidate Selection

**Purpose**: Select the top-k most promising graphs (by GNN prediction) for expensive LLM evaluation, while maintaining a buffer of good candidates across iterations.

**Function**: `select_top_graphs(config, logger, good_graphs_set, predictions)`

**Process**:
1. **Sort Generated Graphs**:
   - Sorts the `predictions` GraphSet by `gnn_score` (descending)
   - Highest predicted scores first

2. **Merge Top Graphs into Good Graphs Set**:
   - Takes the top `top_k_to_keep` graphs (e.g., top 15) from the generated batch
   - Adds them to the persistent `good_graphs_set`
   - Sorts `good_graphs_set` by scores (maintains order)
   - **Important**: `good_graphs_set` accumulates graphs across iterations, so it may contain graphs from previous iterations

3. **Select Graphs for Evaluation**:
   - Takes the top `eval_k_best` graphs (e.g., top 10) from `good_graphs_set`
   - **Removes** these graphs from `good_graphs_set` (they're being evaluated, so don't keep them in the buffer)
   - Creates a new `GraphSet` with just these selected graphs

4. **Enforce Size Limit**:
   - Limits `good_graphs_set` to maximum size of `top_k_to_keep`
   - If it exceeds this size, trims to keep only the top-k graphs

5. **Save State**:
   - Saves the updated `good_graphs_set` to disk (`data/good_graphs_set.pkl`)

6. **Output**:
   - Returns a new `GraphSet` containing exactly `eval_k_best` graphs (typically 10 graphs)
   - These are the graphs that will be evaluated in Step 4

**Metrics Collected**:
- Number of graphs selected for evaluation
- Size of good_graphs_set after selection
- Node type counts in good_graphs_set (statistical summary)

**Example**:
- From 200,000 generated graphs, top 15 are added to `good_graphs_set`
- `good_graphs_set` now has 20 graphs (15 new + 5 from previous iterations)
- Top 10 graphs are selected and removed from `good_graphs_set`
- `good_graphs_set` now has 10 graphs remaining (bottom 5 from previous + 5 of the new top-15)
- Selected graphs have high `gnn_score` values (e.g., 0.85-0.95 range)

**Key Point**: This step implements the core efficiency gain - evaluating only 10 graphs instead of 200,000 represents a 99.995% reduction in evaluations.

---

### STEP 4: LLM Evaluation

**Purpose**: Evaluate the selected graphs by actually running the multi-agent LLM system on mathematical problems and measuring performance.

**Function**: `evaluate_selected_graphs(config, logger, selected_graphs, math_problems)`

**Process**:
1. **LLM Provider Setup**:
   - Sets up the LLM provider (Groq, Ollama, or local) based on configuration
   - Configures API keys, model names, etc.

2. **For Each Selected Graph**:
   a. **Graph to LangGraph Conversion**:
      - Converts the graph structure to a LangGraph executable graph
      - Maps node types to LangGraph node functions (solver_node, validator_node, etc.)
      - Sets up edges to define execution flow
      - Compiles the graph for execution

   b. **Problem Sampling**:
      - Randomly samples `num_eval_problems` problems (e.g., 5 problems) from the math problems dataset
      - Uses the same seed for reproducibility

   c. **For Each Problem**:
      - Creates an initial `AgentState` with the problem text
      - Executes the LangGraph by invoking it with the initial state
      - The graph executes according to its topology:
        - Agents communicate through edges
        - Some agents create new scopes (Decompose creates subproblems, Split creates parallel branches)
        - Final solution is extracted from the state
      - Extracts the solution from the final state (may come from Solver, Combine_all, or Python_solver output)
      - Compares solution to expected answer using `_evaluate_answer()`:
        - For numerical answers: Uses relative error `0.7 * min(a/b, b/a)` for partial credit
        - For exact matches: Score = 1.0
        - For text answers: Uses string matching (exact match = 1.0, contains = 0.7, partial = 0.5, no match = 0.0)
      - Records the score for this problem

   d. **Graph Score Calculation**:
      - Averages all problem scores to get the graph's overall `llm_score`
      - Stores the score in `graph.llm_score`
      - Records evaluation time in `graph.time_evaluating`

3. **Error Handling**:
   - If a graph fails to execute (exception, timeout), assigns `llm_score = 0.0`
   - Logs errors but continues with other graphs

4. **Metrics Calculation**:
   - Calculates RMSE between GNN predictions (`gnn_score`) and actual scores (`llm_score`)
   - Calculates MAE (Mean Absolute Error)
   - Calculates correlation coefficient
   - Tracks per-graph metrics (individual problem scores, execution times)

5. **Output**:
   - Returns the same `GraphSet` with all graphs now having `llm_score` values set
   - All graphs have been evaluated on real mathematical problems

**Metrics Collected**:
- Duration (total evaluation time)
- Number of graphs evaluated
- Best evaluated score (maximum `llm_score`)
- Mean evaluated score
- Worst evaluated score (minimum `llm_score`)
- RMSE between GNN predictions and actual scores
- MAE between predictions and actual scores
- Correlation coefficient
- Per-graph detailed metrics (stored separately, not in CSV)

**Example**:
- Graph with `gnn_score = 0.91` is evaluated on 5 problems
- Problem scores: [1.0, 1.0, 0.87, 1.0, 0.92]
- Average score = 0.958 → `llm_score = 0.958`
- Evaluation time = 12.3 seconds
- RMSE for this iteration = 0.08 (predictions are quite accurate)

**Key Point**: This is the expensive step - each graph evaluation requires multiple LLM API calls and takes seconds to minutes. This is why we only evaluate 10 graphs instead of 200,000.

---

### STEP 5: Data Update

**Purpose**: Add the newly evaluated graphs (with their actual LLM scores) to the training dataset for future GNN training.

**Function**: `update_training_data(config, logger, evaluation_results, training_dataset)`

**Process**:
1. **Add Graphs to Dataset**:
   - Takes all graphs from `evaluation_results` GraphSet
   - Adds them to the `training_dataset` GraphSet
   - Uses `training_dataset.add_graphs()` which appends graphs to the list
   - No sorting is applied (graphs are kept in order added)

2. **Save to Disk**:
   - Saves the updated `training_dataset` to disk as `data/training_dataset.pkl`
   - This ensures persistence across runs (if the pipeline is stopped and restarted, training data is preserved)

3. **Output**:
   - The `training_dataset` is modified in-place
   - Now contains all previously evaluated graphs plus the new ones from this iteration

**Metrics Collected**:
- Number of training samples added (typically equals `eval_k_best`, e.g., 10)
- Total training dataset size (grows by ~10 each iteration)

**Example**:
- Before: `training_dataset` has 120 graphs (from 12 previous iterations)
- Adds 10 newly evaluated graphs
- After: `training_dataset` has 130 graphs
- Each graph in the dataset has both `gnn_score` (prediction) and `llm_score` (actual) set

**Key Point**: The training dataset grows monotonically - it only adds graphs, never removes them. This creates a cumulative learning signal for the GNN.

---

### STEP 6: GNN Retraining

**Purpose**: Retrain the GNN model on the expanded training dataset to improve its ability to predict graph performance.

**Function**: `retrain_gnn_model(config, logger, training_dataset)`

**Process**:
1. **Check Training Data**:
   - Verifies that `training_dataset` is non-empty
   - If empty, returns the model unchanged (can't train on no data)

2. **Create New Model Instance**:
   - Creates a fresh GNN model instance (same architecture, but untrained weights)
   - This ensures clean training without residual effects from previous training

3. **Format Conversion**:
   - Converts all graphs in `training_dataset` to PyTorch Geometric format
   - Creates a list of `Data` or `HeteroData` objects
   - Each object contains:
     - Graph structure (nodes, edges)
     - Node features (type embeddings, structural features)
     - Target label: `llm_score` (the actual performance score)

4. **Training Loop**:
   - Creates a DataLoader with specified `batch_size` (e.g., 32)
   - Runs training for `gnn_epochs` iterations (e.g., 300 epochs)
   - For each epoch:
     - Shuffles data
     - Processes in batches
     - Forward pass: Model predicts scores for all graphs in batch
     - Loss calculation: MSE (Mean Squared Error) between predictions and actual `llm_score` values
     - Backward pass: Updates model weights via Adam optimizer
     - Tracks average loss
   - Logs progress every 10 epochs

5. **Model State**:
   - Marks model as trained (`is_trained = True`)
   - Model weights are updated with learned patterns from all training data

6. **Output**:
   - Returns the retrained model (same object, but with updated weights)
   - This model will be used for predictions in the next iteration's Step 2

**Metrics Collected**:
- Number of training samples used
- Final training loss (MSE)
- Training duration

**Example**:
- Training on 130 graphs
- 300 epochs, batch size 32
- Initial loss: 0.15
- Final loss: 0.08 (model has learned patterns)
- Training time: 45 seconds
- Model is now better at predicting which graph structures will perform well

**Key Point**: Retraining happens every iteration, so the model continuously improves as more evaluation data becomes available. Early iterations may have poor predictions, but later iterations benefit from accumulated knowledge.

---

## Iteration Flow and Loop Back

After Step 6 completes, the iteration is finished. The pipeline:

1. **Collects Metrics**: Combines metrics from all 6 steps into a single dictionary
2. **Saves Iteration Data**: Appends metrics to `logs/analytics/{experiment_name}/all_iterations_data.csv`
3. **Updates State**: Increments iteration counter, updates totals
4. **Loops Back**: Returns to Step 1 for the next iteration

The cycle continues until `max_iterations` is reached (e.g., 40 iterations).

**Important Flow Connections**:
- Step 6 → Step 1: The retrained model is used in the next iteration's Step 2
- Step 5 → Step 6: The updated training_dataset is used for retraining
- Step 3 → Step 3 (next iteration): The `good_graphs_set` persists across iterations, accumulating good candidates
- Step 2 → Step 2 (next iteration): The model used for prediction improves with each retraining

---

## Data Flow Summary

**Graphs Flow**:
1. Generated (Step 1) → 200,000 graphs, no scores
2. Predicted (Step 2) → Same 200,000 graphs, now with `gnn_score`
3. Selected (Step 3) → Top 10 graphs extracted for evaluation
4. Evaluated (Step 4) → Same 10 graphs, now with `llm_score`
5. Added to dataset (Step 5) → 10 graphs added to `training_dataset`
6. Used for training (Step 6) → All graphs in `training_dataset` used to retrain model

**State Persistence**:
- `training_dataset`: Grows monotonically, saved after each iteration
- `good_graphs_set`: Updated each iteration, maintains top-k candidates
- `model`: Retrained each iteration, weights updated (not saved to disk, recreated each iteration)

---

## Key Design Decisions

1. **Why Evaluate Only Top-K?**: LLM evaluation is expensive (time and cost). Evaluating 10 graphs per iteration instead of 200,000 represents a massive efficiency gain (99.995% reduction) while still providing feedback for learning.

2. **Why Retrain Every Iteration?**: The training dataset grows with each iteration. Retraining ensures the model always uses all available data. While computationally expensive, it's still cheaper than evaluating more graphs.

3. **Why Maintain `good_graphs_set`?**: This buffer allows accumulation of promising candidates across iterations. A graph that's #11 in one iteration might be #1 after merging with candidates from other iterations.

4. **Why Start Fresh Model Each Retraining?**: Creating a new model instance ensures clean training without potential issues from incremental updates. The training is fast enough (seconds to minutes) that this is feasible.

5. **Why Filter Duplicates?**: Avoids wasting computation on graphs we've already evaluated. Duplicate checking ensures each unique graph structure is evaluated at most once.

---

## Performance Characteristics

**Per Iteration**:
- Graph generation: ~seconds (fast, random generation)
- GNN prediction: ~seconds to minutes (fast, batch inference)
- Selection: ~milliseconds (fast, sorting)
- LLM evaluation: ~minutes to hours (SLOW, depends on `eval_k_best` and `num_eval_problems`)
- Data update: ~milliseconds (fast, appending)
- Retraining: ~seconds to minutes (moderate, depends on dataset size)

**Total Runtime**: Dominated by Step 4 (LLM evaluation). For 40 iterations with 10 graphs evaluated on 5 problems each: ~40 * 10 * 5 = 2,000 LLM calls.

**Scalability**: 
- Can generate millions of graphs (Step 1)
- Can predict on millions of graphs (Step 2, very fast)
- Limited by LLM evaluation budget (Step 4, expensive)
- Training dataset grows linearly with iterations

---

## Outputs and Artifacts

**During Execution**:
- Console logs: Detailed logging of each step
- Metrics CSV: `logs/analytics/{experiment_name}/all_iterations_data.csv` - One row per iteration with all metrics

**After Completion**:
- Training dataset: `data/training_dataset.pkl` - All evaluated graphs with scores
- Good graphs set: `data/good_graphs_set.pkl` - Top candidate graphs
- Metrics DataFrame: Processed metrics for analysis
- Diagnostics: Visualizations and analysis (if diagnostics are run separately)

---

## Configuration Parameters

Key parameters that control pipeline behavior (from `config/experiment_config.yaml`):

- `num_graphs_per_iteration`: How many graphs to generate (200,000)
- `max_nodes`: Maximum nodes per graph (8)
- `max_depth`: Maximum graph depth (3)
- `top_k_to_keep`: Size of good_graphs_set buffer (15)
- `eval_k_best`: How many graphs to evaluate per iteration (10)
- `num_eval_problems`: Problems per graph evaluation (5)
- `max_iterations`: Total iterations to run (40)
- `gnn_model_type`: Which GNN architecture to use (gcn, gat, hetgat, graphsage)
- `gnn_epochs`: Training epochs per retraining (300)
- `gnn_batch_size`: Batch size for training (32)
- `gnn_learning_rate`: Learning rate for optimizer (0.001)

---

This pipeline represents an active learning approach where the model guides exploration, receives feedback from expensive evaluations, and continuously improves its predictions to guide future exploration more effectively.


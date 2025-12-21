# How Nodes Are Merged in the Scoped Knowledge System

## Overview

The merging process happens through **`Combine_all`** nodes, which collect results from multiple parallel branches (created by `Split` or `Decompose` nodes) and synthesize them into a single result.

## Key Concepts

### 1. **Parallel Execution**
When a `Split` or `Decompose` node creates multiple branches, LangGraph automatically executes nodes in parallel:
- After `Split`: All child nodes solve the **same problem** in parallel
- After `Decompose`: All child nodes solve **different subproblems** in parallel

### 2. **Scope Isolation**
Each branch operates in its own scope:
- `Split` creates scopes like: `root_split_1`, `root_split_2`, `root_split_3`
- `Decompose` creates scopes like: `root_decomp_1`, `root_decomp_2`, `root_decomp_3`

Knowledge is isolated within each scope - nodes in one scope cannot directly access knowledge from another scope.

### 3. **Cross-Scope Knowledge Collection**
The `Combine_all` node is special - it can search **across all scopes** to collect knowledge from its incoming nodes.

## How `Combine_all` Works

### Step 1: Identify Incoming Nodes
When the graph is built, `combine_all_edges` maps each `Combine_all` node to its list of incoming node IDs:

```python
# Example: Combine_all-4 has incoming nodes [2, 3]
combine_all_edges = {
    4: [2, 3]  # Combine_all-4 receives from Solver-2 and Solver-3
}
```

### Step 2: Search Across All Scopes
When `combine_all_node` executes, it:

1. **Iterates through all incoming node IDs**
2. **Searches every scope** to find where each node stored its output
3. **Collects the results**, tracking which scope each came from

```python
for inc_id in incoming:  # e.g., [2, 3]
    # Search all scopes
    for scope, scope_knowledge in scoped_knowledge.items():
        data = scope_knowledge.get(inc_id)
        if data:
            collected_results[inc_id] = {
                "data": str(data),
                "scope": scope  # Track which scope it came from
            }
            break
```

### Step 3: Synthesize Results
All collected results are sent to an LLM for synthesis:

```python
results_text = "\n---\n".join([
    f"Node {nid} (scope: {info['scope']}): {info['data']}" 
    for nid, info in collected_results.items()
])
```

### Step 4: Store Merged Result
The synthesized result is stored in the **root scope**, making it available to downstream nodes.

## Example: Merging After Decompose

```
Original Problem: "Find the sum and product of 8 and 4"

START → Decompose-1 → [Solver-2, Solver-3] → Combine_all-4
                          ↓                      ↓
                   root_decomp_1          root_decomp_2
                   "Calculate sum"        "Calculate product"
                          ↓                      ↓
                   Solution: "12"        Solution: "32"
```

**Execution Flow:**

1. **Decompose-1** breaks the problem:
   - Creates `root_decomp_1` scope → assigns "Calculate sum of 8 and 4" to Solver-2
   - Creates `root_decomp_2` scope → assigns "Calculate product of 8 and 4" to Solver-3

2. **Parallel Execution:**
   - Solver-2 (in `root_decomp_1`) solves: "Calculate sum of 8 and 4" → stores "12" in `root_decomp_1`
   - Solver-3 (in `root_decomp_2`) solves: "Calculate product of 8 and 4" → stores "32" in `root_decomp_2`
   - Both run **in parallel** (LangGraph handles this automatically)

3. **Combine_all-4 Merges:**
   - Searches all scopes for incoming nodes [2, 3]
   - Finds Solver-2's output ("12") in `root_decomp_1`
   - Finds Solver-3's output ("32") in `root_decomp_2`
   - Sends both to LLM for synthesis:
     ```
     Node 2 (scope: root_decomp_1): 12
     Node 3 (scope: root_decomp_2): 32
     ```
   - LLM synthesizes: "Sum is 12, product is 32"
   - Stores merged result in `root` scope

4. **Downstream nodes** (after Combine_all-4) operate in `root` scope and see the merged result.

## Example: Merging After Split

```
START → Split-1 → [Solver-2, Extract_topic-3] → Combine_all-4
                       ↓              ↓
                  root_split_1   root_split_2
                  Same problem   Same problem
                       ↓              ↓
                  Solution: "X"   Topics: "Y"
```

**Execution Flow:**

1. **Split-1** creates parallel branches:
   - Creates `root_split_1` scope → assigns same problem to Solver-2
   - Creates `root_split_2` scope → assigns same problem to Extract_topic-3

2. **Parallel Execution:**
   - Solver-2 (in `root_split_1`) solves the problem → stores solution
   - Extract_topic-3 (in `root_split_2`) analyzes the problem → stores topics
   - Both run **in parallel**

3. **Combine_all-4 Merges:**
   - Searches all scopes for incoming nodes [2, 3]
   - Finds both results from different scopes
   - Synthesizes them into a combined result
   - Stores in `root` scope

## Key Differences: Decompose vs Split

| Aspect | Decompose | Split |
|--------|-----------|-------|
| **Problem Distribution** | Each child gets a **different subproblem** | All children get the **same problem** |
| **Purpose** | Break complex problem into parts | Try multiple approaches in parallel |
| **Merging** | Combine_all **must wait for ALL** children | Combine_all can use **any or all** children |
| **Scopes Created** | `root_decomp_1`, `root_decomp_2`, ... | `root_split_1`, `root_split_2`, ... |

## Important Notes

1. **LangGraph Handles Parallelism**: You don't need to explicitly wait - LangGraph automatically ensures all incoming nodes complete before `Combine_all` executes.

2. **Scope Boundaries**: Regular nodes can only access knowledge in their own scope. Only `Combine_all` can cross scope boundaries to collect knowledge.

3. **Knowledge Flow After Merge**: After `Combine_all` stores results in `root` scope, downstream nodes operate in `root` scope and can access the merged knowledge.

4. **Multiple Combine_all Nodes**: You can have multiple `Combine_all` nodes in a graph, each merging different sets of branches.


# Candidate Pool Visualization Update

## Changes Applied to Slide 7 (Steps 3-4)

Added visual representation of the **Candidate Pool** with GNN scores to better illustrate how graphs are selected and maintained.

### Problem (Before)
- ❌ No visual representation of the candidate pool
- ❌ Not clear what "candidate pool" means
- ❌ GNN scores not shown in the selection process
- ❌ Difficult to understand how graphs are ranked

### Solution (After)
- ✅ **Candidate Pool box** displayed on the right
- ✅ **Individual graphs with GNN scores** shown (e.g., "Graph 1: GNN=0.89")
- ✅ **Visual connection** from "Top 10" to the pool
- ✅ **Clear ranking** - graphs sorted by GNN score
- ✅ Shows pool maintains 10 graphs with their scores

---

## New Visual Layout

```
Left Side (Flow):              Right Side (Pool):
-----------------              ------------------

GNN Predictions               
(1000-2000 graphs)
    ↓ (Sort)
                               ┌─────────────────┐
Top 10 to        ────────→    │ Candidate Pool  │
candidate pool                │                 │
    ↓                         │ Graph 1: 0.89   │
                              │ Graph 2: 0.85   │
Top 5 for                     │ Graph 3: 0.82   │
evaluation                    │      ⋮          │
    ↓                         │ Graph 10: 0.71  │
                              └─────────────────┘
LLM Evaluation
(5 problems/graph)
    ↓
$$$ Cost
```

---

## Detailed Changes

### Candidate Pool Box

**New element added:**
- Large rectangle on the right side
- Light green background (`fill=green!10`)
- Title: "Candidate Pool" at the top
- Contains 10 ranked entries

### Pool Contents (with GNN Scores)

Shows individual graphs with their predicted scores:
```latex
Graph 1: GNN=0.89
Graph 2: GNN=0.85
Graph 3: GNN=0.82
    ⋮
Graph 10: GNN=0.71
```

**Visual elements:**
- Each graph shown as a blue rectangle (`fill=blue!20`)
- GNN score displayed next to each graph
- Vertical dots (⋮) indicating continuation
- Graphs sorted from highest to lowest score

### Arrow Connection

New arrow from "Top 10 to candidate pool" → "Candidate Pool" box
- Shows the flow of graphs into the pool
- Makes the connection explicit

### Updated Positioning

- Main flow shifted slightly left (`x=-0.5`)
- Candidate pool positioned on right (`x=3`)
- Arrow connects the two sections
- Better use of horizontal space

---

## Visual Storytelling

The diagram now tells a complete story:

1. **Input:** 1000-2000 graphs with GNN predictions
2. **Sorting:** Rank by GNN score
3. **Selection:** Top 10 graphs enter candidate pool
4. **Storage:** Pool maintains graphs with their GNN scores (shown visually!)
5. **Evaluation:** Top 5 from pool are evaluated
6. **Cost:** LLM evaluation is expensive

**Key insight:** The GNN scores guide which graphs get into the pool and which get evaluated.

---

## Information Added

### Visual Information
- Candidate pool size: 10 graphs
- Score range: 0.89 (best) to 0.71 (10th)
- Graphs are ordered by GNN prediction
- Pool persists across iterations

### Conceptual Clarity
- Pool acts as a "best graphs so far" storage
- GNN scores determine ranking
- Selection is based on predicted performance
- Only top 5 from pool get expensive LLM evaluation

---

## Technical Details

### TikZ Elements

**Candidate Pool Container:**
```latex
\node[draw, rectangle, fill=green!10, 
      minimum width=2.8cm, minimum height=2cm] at (3, 2.5)
```

**Individual Graph Entries:**
```latex
\node[draw, rectangle, fill=blue!20, 
      minimum width=2.2cm] at (3, y_pos) {Graph N: GNN=score}
```

**Positions:**
- Graph 1: y=3.3 (top, highest score)
- Graph 2: y=2.8
- Graph 3: y=2.3
- Graph 10: y=1.5 (bottom, lowest score)

### Color Scheme
- **Green background:** Candidate pool container
- **Blue rectangles:** Individual graphs
- **Arrow:** Shows flow from selection to pool

### Scale Adjustment
- Changed from `scale=0.65` to `scale=0.7`
- Slightly larger to accommodate new elements
- Better readability

---

## Presentation Benefits

### Clarity
- Audience can now see what the "candidate pool" actually is
- GNN scores are visible and meaningful
- Clear why only some graphs are evaluated

### Teaching Value
- Can point to specific GNN scores
- Shows the filtering mechanism visually
- Demonstrates the ranking system
- Illustrates how GNN predictions guide selection

### Story Flow
1. Generate many graphs
2. GNN predicts scores quickly
3. Keep top 10 in pool (with scores shown!)
4. Evaluate only top 5 with expensive LLMs
5. This is how we save 99.9975% of evaluation cost!

---

## Speaker Notes Updated

Added notes explaining:
- Top 10 stored with their GNN scores
- Pool visualization shows graphs ranked by score
- Examples: 0.89, 0.85, 0.82... down to 0.71
- Pool maintains best graphs across iterations

---

## Key Talking Points

When presenting, emphasize:
- "The candidate pool stores the top 10 graphs with their GNN predicted scores"
- "As you can see on the right, graphs are ranked from 0.89 down to 0.71"
- "We only evaluate the top 5 from this pool with expensive LLM calls"
- "The pool persists across iterations, maintaining the best candidates"
- "This GNN-based filtering is how we achieve massive cost savings"

---

## Visual Comparison

### Before
- Abstract "candidate pool" mentioned but not shown
- No visual representation of ranking
- GNN scores not visible in diagram
- Missing connection to selection process

### After
- ✅ Concrete visual representation of pool
- ✅ Individual graphs with GNN scores displayed
- ✅ Clear ranking from highest to lowest
- ✅ Visual arrow showing flow into pool
- ✅ Demonstrates the filtering mechanism

---

## Files Modified

1. ✅ `graphmind_presentation.tex` - Slide 7 TikZ diagram enhanced
2. ✅ Speaker notes updated with pool explanation
3. ✅ `graphmind_presentation.pdf` - Recompiled (352 KB)

---

## Verification

To view the candidate pool visualization:
```bash
cd presentation
open graphmind_presentation.pdf
```

Navigate to **Slide 7** (Steps 3-4) - you'll see the candidate pool with GNN scores on the right!

---

## Status

✅ **Candidate pool visualized** - Shows 10 graphs with scores  
✅ **GNN scores displayed** - Clear ranking from 0.89 to 0.71  
✅ **Flow connection** - Arrow from selection to pool  
✅ **Professional appearance** - Well-organized layout  
✅ **Ready for presentation** - Clear and informative!  

---

**Updated:** January 11, 2026  
**Requested by:** User - "draw candidate pool to the right... with GNN score"  
**Implementation:** Complete  
**Visual impact:** Excellent - now shows the filtering mechanism clearly! ⭐⭐⭐⭐⭐


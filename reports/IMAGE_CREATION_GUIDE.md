# Guide: Creating Images for Medium Blog Post

This guide covers how to create all the images needed for the Medium blog post.

## Quick Start: Automated Generation

Run the automated script to generate most images:

```bash
python reports/create_blog_images.py
```

This will create images in `reports/blog_images/` directory.

---

## Image Requirements

**Medium Image Guidelines:**
- **Format**: PNG or JPG
- **Resolution**: 300 DPI minimum
- **Width**: 1200-2000 pixels (Medium displays at 1200px max width)
- **Aspect Ratio**: 16:9 or 4:3 works well
- **File Size**: Under 1MB (Medium will compress, but smaller is better)

---

## Image Checklist

### 1. Pipeline Diagram ⭐ (Most Important)

**What it shows**: The 6-step iterative pipeline

**Tools to create**:
- **Option A: Draw.io (Recommended)**
  - Go to https://app.diagrams.net/
  - Use flowchart templates
  - Export as PNG at 300 DPI
  - Save to `reports/blog_images/pipeline_diagram.png`

- **Option B: Python (Mermaid)**
  - Use `graph_builder.py` to generate Mermaid code
  - Paste into https://mermaid.live/
  - Export as PNG
  - Or use Python libraries: `diagrams`, `graphviz`

- **Option C: PowerPoint/Keynote**
  - Create diagram manually
  - Export as PNG (high resolution)

**Style tips**:
- Use consistent colors for each step
- Add arrows showing flow
- Keep it clean and readable
- Label each step clearly

---

### 2. RMSE Trends Over Iterations ✅ (Automated)

**What it shows**: How RMSE decreases over iterations

**How to create**:
```bash
python reports/create_blog_images.py
```
Or manually:
```python
from post_processing.diagnostics import visualize_rmse_trends
from config.settings import Config

config = Config.from_yaml("config/experiment_config.yaml")
visualize_rmse_trends(config, Path("reports/blog_images"), logger)
```

**Output**: `rmse_trends.png` (already blog-post ready at 300 DPI)

---

### 3. Predictions vs Actual Scores ✅ (Automated)

**What it shows**: Scatter plot of GNN predictions vs actual LLM scores

**How to create**:
```bash
python reports/create_blog_images.py
```

**Output**: `predictions_vs_actual.png`

**Customization**: Edit `post_processing/diagnostics.py` → `visualize_predictions_vs_actual()` function

---

### 4. Best Score Progression ✅ (Can be extracted from RMSE plot)

**What it shows**: Best score found over iterations

**How to create**:
- Extract from `all_iterations_data.csv`:
  - Column: `step_4_evaluation_best_evaluated`
  - Plot against `iteration_num`

**Python code**:
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logs/analytics/more-agents-real-llm-v4-bestGCN/all_iterations_data.csv")
plt.figure(figsize=(12, 6))
plt.plot(df['iteration_num'], df['step_4_evaluation_best_evaluated'], 
         marker='o', linewidth=2, markersize=6)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Best Score', fontsize=14)
plt.title('Best Score Progression Over Iterations', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.savefig('reports/blog_images/best_score_progression.png', dpi=300, bbox_inches='tight')
```

---

### 5. Score Distribution Comparison ✅ (Automated)

**What it shows**: Random baseline distribution vs GNN-guided scores

**How to create**:
```bash
python reports/create_blog_images.py
```

**Or manually**:
- Load `distribution_research/results/random_graph_evaluations.csv`
- Plot histogram of `average_score`
- Overlay Beta/GMM distribution fits

**Output**: `distribution_comparison.png`

---

### 6. Top-Performing Graph Structures ✅ (Automated)

**What it shows**: Visualizations of best-performing graph topologies

**How to create**:
```bash
python reports/create_blog_images.py
```

**Output**: `best_graphs/` directory with individual graph visualizations

**Customization**: Edit `post_processing/diagnostics.py` → `visualize_best_graphs()` function

---

## Manual Creation Tips

### For Python/Matplotlib Plots:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for Medium
plt.style.use('seaborn-v0_8-whitegrid')  # Clean, professional
sns.set_palette("husl")  # Colorful but readable

# High DPI for Medium
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.facecolor'] = 'white'

# Create plot
fig, ax = plt.subplots(figsize=(12, 8))
# ... your plotting code ...

# Save
plt.savefig('output.png', dpi=300, bbox_inches='tight', facecolor='white')
```

### For Graph Structure Visualizations:

**Using NetworkX** (you already have this):
```python
import networkx as nx
import matplotlib.pyplot as plt

# Create graph
G = nx.DiGraph()
# ... add nodes and edges ...

# Visualize
pos = nx.spring_layout(G, k=1, iterations=50)
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=1000, font_size=10, arrows=True, 
        edge_color='gray', width=2)

plt.savefig('graph_structure.png', dpi=300, bbox_inches='tight', facecolor='white')
```

### For Pipeline Diagrams:

**Using Draw.io**:
1. Go to https://app.diagrams.net/
2. Choose "Flowchart" template
3. Create boxes for each step (1-6)
4. Add arrows between steps
5. Use consistent colors:
   - Step 1 (Generation): Blue
   - Step 2 (Prediction): Green
   - Step 3 (Selection): Yellow
   - Step 4 (Evaluation): Red
   - Step 5 (Update): Purple
   - Step 6 (Retrain): Orange
6. Export → PNG → 300 DPI

**Using Python `diagrams` library**:
```bash
pip install diagrams
```

```python
from diagrams import Diagram, Cluster
from diagrams.onprem.client import Users
from diagrams.onprem.compute import Server

with Diagram("Pipeline", filename="pipeline", show=False, direction="LR"):
    gen = Server("1. Generate")
    pred = Server("2. Predict")
    sel = Server("3. Select")
    eval = Server("4. Evaluate")
    update = Server("5. Update")
    retrain = Server("6. Retrain")
    
    gen >> pred >> sel >> eval >> update >> retrain
    retrain >> gen  # Loop back
```

---

## Image Organization

Save all images to: `reports/blog_images/`

**File naming convention**:
- `pipeline_diagram.png` - Main pipeline overview
- `rmse_trends.png` - RMSE improvement
- `predictions_vs_actual.png` - Scatter plot
- `best_score_progression.png` - Score over time
- `distribution_comparison.png` - Random vs GNN
- `best_graphs/` - Directory with top graph visualizations

---

## Quality Checklist

Before using images in Medium:

- [ ] Resolution is 300 DPI or higher
- [ ] Text is readable at Medium's display size
- [ ] Colors are consistent across images
- [ ] File size is reasonable (< 1MB)
- [ ] White background (or transparent PNG)
- [ ] No pixelation when zoomed
- [ ] All labels and legends are clear

---

## Quick Reference: Image Sizes for Medium

- **Full-width images**: 1200px width (Medium max)
- **Half-width images**: 600px width
- **Small inline images**: 300-400px width

**Aspect ratios that work well**:
- 16:9 (1920x1080) - Wide, good for trends
- 4:3 (1600x1200) - Balanced
- 1:1 (1200x1200) - Square, good for distributions

---

## Troubleshooting

**Images look blurry in Medium?**
- Export at 300 DPI minimum
- Use PNG format (better than JPG for diagrams)
- Check that Medium isn't compressing too much

**Text is too small?**
- Increase font sizes in matplotlib: `plt.rcParams['font.size'] = 14`
- Use bold fonts for titles: `fontweight='bold'`

**Colors look washed out?**
- Use `seaborn` color palettes
- Avoid very light colors
- Test on white background

---

## Next Steps

1. Run `python reports/create_blog_images.py` to generate automated images
2. Create pipeline diagram manually (Draw.io recommended)
3. Review all images and adjust as needed
4. Add image references to `final_report.md`


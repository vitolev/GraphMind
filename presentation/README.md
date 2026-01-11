# GraphMind Presentation

This folder contains the 9-minute Beamer presentation for the GraphMind project.

## Files

- `graphmind_presentation.tex` - Main Beamer presentation file
- `create_gnn_visualization.py` - Python script to generate GNN architecture diagrams
- `figures/` - Generated figures for the presentation
- `Makefile` - Easy compilation commands

## Compilation

### Quick Start
```bash
make all
```

This will:
1. Generate the visualization figures
2. Compile the LaTeX presentation
3. Open the PDF

### Individual Commands

Generate figures only:
```bash
make figures
```

Compile presentation only:
```bash
make presentation
```

Clean build artifacts:
```bash
make clean
```

## Presentation Structure (9 minutes)

1. **Title Slide** (30s)
2. **Problem Introduction** (60s) - The topology search challenge
3. **Multi-Agent System** (60s) - Agent types and graph structure
4. **Pipeline Overview** (60s) - 6-step iterative process
5. **Graph Generation** (60s) - Step 1 in detail
6. **GNN Prediction** (90s) - Step 2 with virtual node explanation
7. **Selection & Evaluation** (60s) - Steps 3-4
8. **Update & Retrain** (60s) - Steps 5-6
9. **GNN Architecture** (60s) - Models and hyperparameters
10. **Results: GNN Performance** (60s) - Hyperparameter search results
11. **Results: Distribution** (60s) - GNN-guided vs Random
12. **Example Graphs** (60s) - Good vs bad topologies
13. **Future Work** (60s) - Challenges and improvements
14. **Conclusion** (30s) - Summary and questions

**Total: ~13 slides, 9 minutes**

## Speaker Notes

The presentation includes detailed speaker notes for each slide. To view them during presentation:
- Compile with notes: The presentation is set up to show notes on second screen
- Or compile without notes by commenting out the line in the .tex file:
  ```latex
  % \setbeameroption{show notes on second screen=right}
  ```

## Figures Used

The presentation uses several figures from the project:
- Generated GNN architecture diagrams (from `create_gnn_visualization.py`)
- Results from `../reports/blog_images/random_vs_gnn_comparison.png`
- Custom TikZ diagrams for pipeline steps

## Tips for Presentation

1. **Practice timing** - Aim for 8-9 minutes to leave time for questions
2. **Focus on visuals** - The slides are image-heavy by design
3. **Key messages**:
   - Problem: Millions of possible topologies, expensive to evaluate
   - Solution: GNN surrogate models predict performance
   - Result: 99.9975% cost reduction
4. **Emphasize** the pipeline and GNN architecture (core contributions)
5. **Be ready to explain** virtual node technique and message passing

## Customization

To adjust the presentation:
- Change colors in TikZ diagrams by modifying the color definitions
- Adjust timing by focusing more/less on certain slides
- Add more examples from `../reports/good-bad-graphs/`
- Update results with latest numbers

## Requirements

- LaTeX with Beamer
- Python 3.x with matplotlib (for figure generation)
- Virtual environment: `../graphmind-venv/`

## Contact

Vito Levstik and Gal Zmazek  
University of Ljubljana


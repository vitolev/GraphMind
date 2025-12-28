# Quick Start: Stanford-Style Visualizations

## Overview

This guide helps you create publication-quality visualizations inspired by Stanford CS224W and other top-tier academic publications.

## What's Included

1. **VISUALIZATION_GUIDE.md** - Comprehensive guide with design principles, code examples, and best practices
2. **create_stanford_visualizations.py** - Ready-to-run script that generates professional figures

## Quick Start

### 1. Generate Default Visualizations

```bash
python reports/create_stanford_visualizations.py
```

This creates:
- `pipeline_flow_stanford.png` - 6-step pipeline diagram
- `graph_simple.png` - Simple graph topology example
- `graph_complex.png` - Complex graph topology example  
- `graph_python_validator.png` - Python solver with validation example
- `gnn_architecture_stanford.png` - GNN architecture diagram

All saved to `reports/blog_images/`

### 2. Customize for Your Needs

Edit `create_stanford_visualizations.py` to:
- Change colors (using `STANFORD_COLORS` dictionary)
- Add your own graph topologies
- Modify layouts and styles
- Create additional visualization types

## Key Design Principles

1. **Clean, Minimalist**: White backgrounds, high contrast, ample whitespace
2. **Professional Colors**: ColorBrewer palettes (colorblind-friendly)
3. **High Resolution**: 300 DPI minimum (600 for print)
4. **Clear Typography**: Sans-serif fonts (Arial, Helvetica)
5. **Consistent Styling**: Same colors/patterns across all figures

## Color Palette

The script uses a Stanford-inspired color scheme:

- **Teal** (`#66C2A5`) - Steps, processes
- **Blue-Purple** (`#8DA0CB`) - Arrows, inputs
- **Orange-Red** (`#FC8D62`) - Highlights, outputs
- **Light Gray** (`#E8E8E8`) - Background elements
- **Dark Gray** (`#1F1F1F`) - Text, borders

## Adding Custom Visualizations

### Example: Adding a Score Comparison Chart

```python
def create_score_comparison_chart(data, output_path):
    """Add this to create_stanford_visualizations.py"""
    setup_stanford_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Your plotting code here
    # ...
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
```

## Tips for Publication

1. **Export Settings**:
   - Always use `dpi=300` (or 600 for print)
   - Use `bbox_inches='tight'` to avoid clipping
   - Set `facecolor='white'` for clean backgrounds

2. **File Formats**:
   - **PNG**: Good for web, presentations (use 300 DPI)
   - **PDF**: Best for papers (vector graphics)
   - **SVG**: For further editing in Illustrator/Inkscape

3. **Final Touches**:
   - Review all labels for clarity
   - Ensure legends are clear and positioned well
   - Check color contrast (especially for colorblind readers)
   - Verify all text is readable at final size

## Integration with Your Report

These visualizations are designed to be:
- **Blog-ready**: Sized for Medium/blog posts
- **Paper-ready**: High DPI, professional styling
- **Presentation-ready**: Clear, readable at distance

## Next Steps

1. Run the script to see the default visualizations
2. Review `VISUALIZATION_GUIDE.md` for detailed explanations
3. Customize colors and layouts to match your brand
4. Add data-driven visualizations using your experiment results
5. Create animations (optional, using Manim) for presentations

## Troubleshooting

**Issue**: Graphviz layout not working
- **Solution**: Install Graphviz (`brew install graphviz` on Mac, or `apt-get install graphviz` on Linux)
- The script will automatically fall back to spring layout if Graphviz is unavailable

**Issue**: Fonts look different
- **Solution**: Ensure Arial/Helvetica are installed, or modify `setup_stanford_style()` to use available fonts

**Issue**: Images are blurry
- **Solution**: Always use `dpi=300` when saving, and check your viewer's zoom level

## Examples in Academic Publications

For inspiration, check:
- Stanford CS224W course materials
- NeurIPS, ICML, ICLR papers (especially graph learning papers)
- Nature/Science visualizations for clean design principles

Happy visualizing! 🎨


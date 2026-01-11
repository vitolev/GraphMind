# Makefile for GraphMind Presentation

# Variables
VENV = ../graphmind-venv
PYTHON = $(VENV)/bin/python
TEX_FILE = graphmind_presentation.tex
PDF_FILE = graphmind_presentation.pdf
VIZ_SCRIPT = create_gnn_visualization.py

.PHONY: all figures presentation clean open help

# Default target
all: figures presentation open

# Generate visualization figures
figures:
	@echo "Generating visualization figures..."
	@$(PYTHON) $(VIZ_SCRIPT)
	@echo "Figures generated successfully!"

# Compile the presentation
presentation:
	@echo "Compiling presentation..."
	@pdflatex -interaction=nonstopmode $(TEX_FILE) > /dev/null
	@pdflatex -interaction=nonstopmode $(TEX_FILE) > /dev/null
	@echo "Presentation compiled successfully!"

# Open the PDF
open:
	@if [ -f $(PDF_FILE) ]; then \
		echo "Opening presentation..."; \
		open $(PDF_FILE); \
	else \
		echo "PDF not found. Run 'make presentation' first."; \
	fi

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb
	@rm -f figures/*.png
	@echo "Clean complete!"

# Deep clean (including PDF)
deepclean: clean
	@rm -f $(PDF_FILE)
	@echo "Deep clean complete!"

# Help
help:
	@echo "GraphMind Presentation Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  all          - Generate figures, compile presentation, and open PDF (default)"
	@echo "  figures      - Generate visualization figures only"
	@echo "  presentation - Compile LaTeX presentation only"
	@echo "  open         - Open the compiled PDF"
	@echo "  clean        - Remove build artifacts (keep PDF)"
	@echo "  deepclean    - Remove all generated files including PDF"
	@echo "  help         - Show this help message"


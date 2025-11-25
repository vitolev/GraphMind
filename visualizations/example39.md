---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	python_solver_1(python_solver_1)
	__end__([<p>__end__</p>]):::last
	__start__ --> python_solver_1;
	python_solver_1 --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

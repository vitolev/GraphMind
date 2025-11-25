---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	explain_1(explain_1)
	extract_topic_2(extract_topic_2)
	python_solver_3(python_solver_3)
	__end__([<p>__end__</p>]):::last
	__start__ --> explain_1;
	explain_1 --> extract_topic_2;
	extract_topic_2 --> python_solver_3;
	python_solver_3 --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

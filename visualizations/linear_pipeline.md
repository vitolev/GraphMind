---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	extract_topic_1(extract_topic_1)
	solver_2(solver_2)
	validator_3(validator_3)
	__end__([<p>__end__</p>]):::last
	__start__ --> extract_topic_1;
	extract_topic_1 --> solver_2;
	solver_2 --> validator_3;
	validator_3 --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

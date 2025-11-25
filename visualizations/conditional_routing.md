---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	validator_1(validator_1)
	true_pass_2(true_pass_2)
	false_pass_3(false_pass_3)
	solver_4(solver_4)
	solver_5(solver_5)
	__end__([<p>__end__</p>]):::last
	__start__ --> validator_1;
	false_pass_3 --> solver_5;
	true_pass_2 --> solver_4;
	validator_1 --> false_pass_3;
	validator_1 --> true_pass_2;
	solver_4 --> __end__;
	solver_5 --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

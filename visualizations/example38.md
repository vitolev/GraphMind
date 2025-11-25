---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	explain_1(explain_1)
	solver_2(solver_2)
	validator_3(validator_3)
	true_pass_4(true_pass_4)
	combine_any_5(combine_any_5)
	false_pass_7(false_pass_7)
	python_solver_8(python_solver_8)
	__end__([<p>__end__</p>]):::last
	__start__ --> explain_1;
	explain_1 --> solver_2;
	false_pass_7 --> python_solver_8;
	python_solver_8 --> combine_any_5;
	solver_2 --> validator_3;
	true_pass_4 --> combine_any_5;
	validator_3 --> false_pass_7;
	validator_3 --> true_pass_4;
	combine_any_5 --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

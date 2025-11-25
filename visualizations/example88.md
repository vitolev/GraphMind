---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	solver_1(solver_1)
	validator_2(validator_2)
	true_pass_3(true_pass_3)
	combine_any_4(combine_any_4)
	python_solver_5(python_solver_5)
	false_pass_7(false_pass_7)
	python_solver_8(python_solver_8)
	__end__([<p>__end__</p>]):::last
	__start__ --> solver_1;
	combine_any_4 --> python_solver_5;
	false_pass_7 --> python_solver_8;
	python_solver_8 --> combine_any_4;
	solver_1 --> validator_2;
	true_pass_3 --> combine_any_4;
	validator_2 --> false_pass_7;
	validator_2 --> true_pass_3;
	python_solver_5 --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

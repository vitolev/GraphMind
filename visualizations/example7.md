---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	python_solver_1(python_solver_1)
	validator_2(validator_2)
	true_pass_3(true_pass_3)
	combine_any_4(combine_any_4)
	solver_5(solver_5)
	validator_6(validator_6)
	true_pass_7(true_pass_7)
	combine_any_8(combine_any_8)
	false_pass_11(false_pass_11)
	python_solver_12(python_solver_12)
	false_pass_13(false_pass_13)
	decompose_2_14(decompose_2_14)
	python_solver_15(python_solver_15)
	combine_all_16(combine_all_16)
	solver_17(solver_17)
	python_solver_18(python_solver_18)
	__end__([<p>__end__</p>]):::last
	__start__ --> python_solver_1;
	combine_all_16 --> solver_17;
	combine_any_4 --> solver_5;
	decompose_2_14 --> python_solver_15;
	decompose_2_14 --> python_solver_18;
	false_pass_11 --> python_solver_12;
	false_pass_13 --> decompose_2_14;
	python_solver_1 --> validator_2;
	python_solver_12 --> combine_any_8;
	python_solver_15 --> combine_all_16;
	python_solver_18 --> combine_all_16;
	solver_17 --> combine_any_4;
	solver_5 --> validator_6;
	true_pass_3 --> combine_any_4;
	true_pass_7 --> combine_any_8;
	validator_2 --> false_pass_13;
	validator_2 --> true_pass_3;
	validator_6 --> false_pass_11;
	validator_6 --> true_pass_7;
	combine_any_8 --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

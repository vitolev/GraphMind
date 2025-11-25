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
	decompose_2_5(decompose_2_5)
	python_solver_6(python_solver_6)
	combine_all_7(combine_all_7)
	python_solver_8(python_solver_8)
	solver_10(solver_10)
	false_pass_11(false_pass_11)
	decompose_2_12(decompose_2_12)
	python_solver_13(python_solver_13)
	combine_all_14(combine_all_14)
	solver_15(solver_15)
	python_solver_16(python_solver_16)
	__end__([<p>__end__</p>]):::last
	__start__ --> solver_1;
	combine_all_14 --> solver_15;
	combine_all_7 --> python_solver_8;
	combine_any_4 --> decompose_2_5;
	decompose_2_12 --> python_solver_13;
	decompose_2_12 --> python_solver_16;
	decompose_2_5 --> python_solver_6;
	decompose_2_5 --> solver_10;
	false_pass_11 --> decompose_2_12;
	python_solver_13 --> combine_all_14;
	python_solver_16 --> combine_all_14;
	python_solver_6 --> combine_all_7;
	solver_1 --> validator_2;
	solver_10 --> combine_all_7;
	solver_15 --> combine_any_4;
	true_pass_3 --> combine_any_4;
	validator_2 --> false_pass_11;
	validator_2 --> true_pass_3;
	python_solver_8 --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

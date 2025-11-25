---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	explain_1(explain_1)
	python_solver_2(python_solver_2)
	validator_3(validator_3)
	true_pass_4(true_pass_4)
	combine_any_5(combine_any_5)
	decompose_4_6(decompose_4_6)
	solver_7(solver_7)
	combine_all_8(combine_all_8)
	python_solver_9(python_solver_9)
	python_solver_11(python_solver_11)
	solver_12(solver_12)
	solver_13(solver_13)
	false_pass_14(false_pass_14)
	extract_topic_15(extract_topic_15)
	split_16(split_16)
	python_solver_17(python_solver_17)
	combine_all_18(combine_all_18)
	python_solver_19(python_solver_19)
	python_solver_20(python_solver_20)
	__end__([<p>__end__</p>]):::last
	__start__ --> explain_1;
	combine_all_18 --> python_solver_19;
	combine_all_8 --> python_solver_9;
	combine_any_5 --> decompose_4_6;
	decompose_4_6 --> python_solver_11;
	decompose_4_6 --> solver_12;
	decompose_4_6 --> solver_13;
	decompose_4_6 --> solver_7;
	explain_1 --> python_solver_2;
	extract_topic_15 --> split_16;
	false_pass_14 --> extract_topic_15;
	python_solver_11 --> combine_all_8;
	python_solver_17 --> combine_all_18;
	python_solver_19 --> combine_any_5;
	python_solver_2 --> validator_3;
	python_solver_20 --> combine_all_18;
	solver_12 --> combine_all_8;
	solver_13 --> combine_all_8;
	solver_7 --> combine_all_8;
	split_16 --> python_solver_17;
	split_16 --> python_solver_20;
	true_pass_4 --> combine_any_5;
	validator_3 --> false_pass_14;
	validator_3 --> true_pass_4;
	python_solver_9 --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

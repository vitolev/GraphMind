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
	true_pass_4(true_pass_4)
	combine_any_5(combine_any_5)
	python_solver_6(python_solver_6)
	validator_7(validator_7)
	true_pass_8(true_pass_8)
	combine_any_9(combine_any_9)
	false_pass_12(false_pass_12)
	python_solver_13(python_solver_13)
	false_pass_14(false_pass_14)
	extract_topic_15(extract_topic_15)
	explain_16(explain_16)
	python_solver_17(python_solver_17)
	__end__([<p>__end__</p>]):::last
	__start__ --> extract_topic_1;
	combine_any_5 --> python_solver_6;
	explain_16 --> python_solver_17;
	extract_topic_1 --> solver_2;
	extract_topic_15 --> explain_16;
	false_pass_12 --> python_solver_13;
	false_pass_14 --> extract_topic_15;
	python_solver_13 --> combine_any_9;
	python_solver_17 --> combine_any_5;
	python_solver_6 --> validator_7;
	solver_2 --> validator_3;
	true_pass_4 --> combine_any_5;
	true_pass_8 --> combine_any_9;
	validator_3 --> false_pass_14;
	validator_3 --> true_pass_4;
	validator_7 --> false_pass_12;
	validator_7 --> true_pass_8;
	combine_any_9 --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

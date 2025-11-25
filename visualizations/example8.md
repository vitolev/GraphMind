---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	split_1(split_1)
	solver_2(solver_2)
	combine_all_3(combine_all_3)
	decompose_4_4(decompose_4_4)
	solver_5(solver_5)
	combine_all_6(combine_all_6)
	python_solver_7(python_solver_7)
	solver_9(solver_9)
	solver_10(solver_10)
	python_solver_11(python_solver_11)
	solver_12(solver_12)
	__end__([<p>__end__</p>]):::last
	__start__ --> split_1;
	combine_all_3 --> decompose_4_4;
	combine_all_6 --> python_solver_7;
	decompose_4_4 --> python_solver_11;
	decompose_4_4 --> solver_10;
	decompose_4_4 --> solver_5;
	decompose_4_4 --> solver_9;
	python_solver_11 --> combine_all_6;
	solver_10 --> combine_all_6;
	solver_12 --> combine_all_3;
	solver_2 --> combine_all_3;
	solver_5 --> combine_all_6;
	solver_9 --> combine_all_6;
	split_1 --> solver_12;
	split_1 --> solver_2;
	python_solver_7 --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

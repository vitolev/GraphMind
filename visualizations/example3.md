---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	split_1(split_1)
	decompose_3_2(decompose_3_2)
	python_solver_3(python_solver_3)
	combine_all_4(combine_all_4)
	python_solver_5(python_solver_5)
	combine_all_6(combine_all_6)
	decompose_3_7(decompose_3_7)
	python_solver_8(python_solver_8)
	combine_all_9(combine_all_9)
	python_solver_10(python_solver_10)
	solver_12(solver_12)
	python_solver_13(python_solver_13)
	solver_14(solver_14)
	python_solver_15(python_solver_15)
	decompose_4_16(decompose_4_16)
	solver_17(solver_17)
	combine_all_18(combine_all_18)
	python_solver_19(python_solver_19)
	python_solver_20(python_solver_20)
	solver_21(solver_21)
	python_solver_22(python_solver_22)
	__end__([<p>__end__</p>]):::last
	__start__ --> split_1;
	combine_all_18 --> python_solver_19;
	combine_all_4 --> python_solver_5;
	combine_all_6 --> decompose_3_7;
	combine_all_9 --> python_solver_10;
	decompose_3_2 --> python_solver_15;
	decompose_3_2 --> python_solver_3;
	decompose_3_2 --> solver_14;
	decompose_3_7 --> python_solver_13;
	decompose_3_7 --> python_solver_8;
	decompose_3_7 --> solver_12;
	decompose_4_16 --> python_solver_20;
	decompose_4_16 --> python_solver_22;
	decompose_4_16 --> solver_17;
	decompose_4_16 --> solver_21;
	python_solver_13 --> combine_all_9;
	python_solver_15 --> combine_all_4;
	python_solver_19 --> combine_all_6;
	python_solver_20 --> combine_all_18;
	python_solver_22 --> combine_all_18;
	python_solver_3 --> combine_all_4;
	python_solver_5 --> combine_all_6;
	python_solver_8 --> combine_all_9;
	solver_12 --> combine_all_9;
	solver_14 --> combine_all_4;
	solver_17 --> combine_all_18;
	solver_21 --> combine_all_18;
	split_1 --> decompose_3_2;
	split_1 --> decompose_4_16;
	python_solver_10 --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

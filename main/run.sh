bash
#!/bin/bash

# build data ==========
#python doc2dia_zh.py --source_path="/home/build_data/refGPT/RefGPT-Dataset-V1-CN.json" --prompt="/home/SDXX/build_data/ww_data/doc2dia_chinese_basePrompt.json" --output_path="/home/build_data/refGPT/doc2dia_ref_base(500.json" --name=0

#python doc2dia_zh.py --source_path="/home/build_data/refGPT/RefGPT-Dataset-V1-CN.json" --prompt="/home/SDXX/build_data/ww_data/doc2dia_ww_prompt.json" --output_path="/home/build_data/refGPT/doc2dia_ref_my(500.json" --name=1

#python doc2dia_zh.py --source_path="/home/build_data/ww/museum_use.json" --prompt="/home/SDXX/build_data/ww_data/doc2dia_ww_prompt.json" --output_path="/home/build_data/ww/doc2dia_ww_my(1k.json" --label=1 --name=3

#python doc2dia_zh.py --source_path="/home/build_data/ww/museum_use.json" --prompt="/home/SDXX/build_data/ww_data/doc2dia_chinese_basePrompt.json" --output_path="/home/build_data/ww/doc2dia_ww_base(2k.json" --label=1 --name=2

#python doc2dia_zh.py --source_path="/home/build_data/squad_v2/squad_v2_train.json" --prompt="/home/SDXX/build_data/eng_data/doc2dia_eng_basePrompt.json" --output_path="/home/build_data/eng_data/doc2dia_eng_base(2k.json" --label=2 --name=4
#
python doc2dia_zh.py --source_path="/home/build_data/squad_v2/squad_v2_train.json" --prompt="/home/SDXX/build_data/eng_data/doc2dia_eng_prompt.json" --output_path="/home/build_data/squad_v2/doc2dia_eng_my(1k.json" --label=2 --name=5

# eval data ==========
#python dia2eval_zh.py --source_path="/home/build_data/refGPT/doc2dia_ref_base(500).json" --output_path="/home/build_data/refGPT/dia2eval_ref_base(500.json" --name=0
#
#python dia2eval_zh.py --source_path="/home/build_data/refGPT/doc2dia_ref_my(500).json" --output_path="/home/build_data/refGPT/dia2eval_ref_my(500.json" --name=1
#
#python dia2eval_zh.py --source_path="/home/build_data/ww/doc2dia_ww_base(500).json" --output_path="/home/build_data/ww/dia2eval_ww_base(300.json" --name=2
#
#python dia2eval_zh.py --source_path="/home/build_data/ww/doc2dia_ww_my(500).json" --output_path="/home/build_data/ww/dia2eval_ww_my(300.json" --name=3

# score data ==========
#python eval2score.py --source_path="/home/build_data/refGPT/dia2eval_ref_base(500).json" --output_path="/home/build_data/refGPT/score_ref_base(500).json"
#
#python eval2score.py --source_path="/home/build_data/refGPT/dia2eval_ref_my(500).json" --output_path="/home/build_data/refGPT/score_ref_my(500).json"
#
#python eval2score.py --source_path="/home/build_data/ww/dia2eval_ww_base(500).json" --output_path="/home/build_data/ww/score_ww_base(500).json"
#
#python eval2score.py --source_path="/home/build_data/ww/dia2eval_ww_my(500).json" --output_path="/home/build_data/ww/score_ww_my(500).json"

#python doc2dia_zh.py --source_path="/home/build_data/ww/museum_use.json" --prompt="/home/SDXX/build_data/ww_data/doc2dia_chinese_basePrompt.json" --output_path="/home/build_data/ww/doc2dia_ww_base(500.json" --label=1 --name=2
#python dia2eval_zh.py --source_path="/home/build_data/ww/doc2dia_ww_base(500).json" --output_path="/home/build_data/ww/dia2eval_ww_base.json" --name=2

#python doc2dia_zh.py --source_path="/home/build_data/ww/museum_use.json" --prompt="/home/SDXX/build_data/ww_data/doc2dia_ww_prompt.json" --output_path="/home/build_data/ww/doc2dia_ww_my(500.json" --label=1 --name=3
#python dia2eval_zh.py --source_path="/home/build_data/ww/doc2dia_ww_my(500).json" --output_path="/home/build_data/ww/dia2eval_ww_my.json" --name=3


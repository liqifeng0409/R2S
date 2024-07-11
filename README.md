# R2S
This repository contains the code and data for our paper Raw Text is All you Need: 
Knowledge-intensive Multi-turn Instruction Tuning for Large Language Model 
([R2S](http://arxiv.org/abs/2407.03040))

## Installation
```commandline
pip -r install requirements.txt
```
## Raw Document Data
data path `./train_data/raw_document.zip`  
-- museum_use.json  
-- RefGPT-Dataset-V1-CN.json  
-- squadV2.json  

## Prompts
prompt path `./prompt`  
-- doc2dia: Generate dialogs from documents  
-- basePrompt: Generate dialogs directly from documents  
-- CoTPrompt: Generate dialogs from documents using the CoT approach  
-- CoDPrompt: Generate dialogs from documents using the CoD approach (Ours)  
-- eval_prompt: Evaluating the dialog  

## Data build code  
promppt path `./main`  
-- doc2dia.py: Generate dialogs from documents  
-- dia2eval.py: Evaluating the dialog  
-- eval2score.py: Count eval's scores  
-- auto.py: Automated evaluation indicator: coverage  
-- run.sh: Run scripts for dialog data generation, evaluation, statistics  

## SFT data
data path `./dial_build_data/dial_build_sft_data.zip`  
-- ww: museum data  
-- en: squadV2 data  
-- zh: RefGPTdata

## Model  
Our model are availble [here]

## Train
To train our model on the preprocessed data, please run following code:
```commandline

```

## Inference
To reproduce our result in the paper, please run following code:
```

```

## Evaluation result
data path `./eval_result`  
-- dial-build_model_eval: Data Building Models  
-- dial_model_eval: dialog model  
-- 

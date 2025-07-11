#!/bin/bash

model=qwen2.5-1.5b
path=/aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/huggingface/Qwen2.5-1.5B

mkdir -p outputs/$model

python src/qwen.py \
    --dataset_name triviaqa --dataset_path /aistor/sjtu/hpc_stor01/home/luoyijie/data/wikipedia/triviaqa/val.jsonl \
    --template_name wiki_main_topics \
    --output_path outputs/$model/triviaqa_wiki_main_topics.jsonl \
    --model_name_or_path $path \
    --topk 1 \
    --resume_from_raw_results outputs/$model/triviaqa_wiki_main_topics.jsonl \
    --data_parallel_size 8 --batch_size 16 > logs/${model}_triviaqa_wiki_main_topics.log 2>&1 &
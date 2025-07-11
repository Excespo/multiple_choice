#!/bin/bash

mkdir -p outputs/qwen2.5-14b

python src/qwen.py \
    --dataset_name triviaqa --dataset_path /aistor/sjtu/hpc_stor01/home/luoyijie/data/wikipedia/triviaqa/val.jsonl \
    --template_name wiki_main_topics \
    --output_path outputs/qwen2.5-14b/triviaqa_wiki_main_topics.jsonl \
    --model_name_or_path /aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/huggingface/Qwen2.5-14B-Instruct \
    --data_parallel_size 8 --batch_size 16 > logs/qwen2.5-14b_triviaqa_wiki_main_topics.log 2>&1 &
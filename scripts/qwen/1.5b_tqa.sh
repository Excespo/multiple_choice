#!/bin/bash

python src/qwen.py \
    --dataset_name triviaqa --dataset_path /aistor/sjtu/hpc_stor01/home/luoyijie/data/wikipedia/triviaqa/val.jsonl \
    --template_name wiki_main_topics \
    --output_path outputs/qwen2.5-1.5b/triviaqa_wiki_main_topics.jsonl \
    --model_name_or_path /aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/huggingface/Qwen2.5-1.5B-Instruct \
    --data_parallel_size 8 --batch_size 16 > logs/qwen2.5-1.5b_triviaqa_wiki_main_topics.log 2>&1 &
#!/bin/bash

mkdir -p outputs/qwen3-8b

python src/qwen.py \
    --dataset_name triviaqa \
    --dataset_path /aistor/sjtu/hpc_stor01/home/luoyijie/data/wikipedia/triviaqa/raw/unfiltered_nocontext_validation.jsonl \
    --template_name wiki_main_topics \
    --output_path outputs/qwen3-8b/triviaqa_wiki_main_topics.jsonl \
    --model_name_or_path /aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/huggingface/Qwen3-8B \
    --data_parallel_size 8 --batch_size 64 > logs/qwen3-8b_triviaqa_wiki_main_topics.log 2>&1 &
#!/bin/bash

mkdir -p outputs/qwen2.5-7b

python src/run_qwen_eval.py \
    --dataset_name triviaqa \
    --dataset_path /aistor/sjtu/hpc_stor01/home/luoyijie/data/wikipedia/triviaqa/raw/unfiltered_nocontext_validation.jsonl \
    --template_name wiki_main_topics \
    --output_path outputs/qwen2.5-7b/triviaqa_wiki_main_topics.jsonl \
    --model_name_or_path /aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/huggingface/Qwen2.5-7B-Instruct \
    --data_parallel_size 8 --batch_size 16 > logs/qwen2.5-7b_triviaqa_wiki_main_topics.log 2>&1 &

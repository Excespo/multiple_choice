#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

mkdir -p outputs/qwen2.5-14b

python src/qwen.py \
    --dataset_name natural_questions \
    --dataset_path /aistor/sjtu/hpc_stor01/home/luoyijie/data/wikipedia/nq_open/raw/nq_open_validation.jsonl \
    --template_name wiki_main_topics \
    --output_path outputs/qwen2.5-14b/nq_open_wiki_main_topics.jsonl \
    --model_name_or_path /aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/huggingface/Qwen2.5-14B-Instruct \
    --data_parallel_size 8 --batch_size 64 > logs/qwen2.5-14b_nq_open_wiki_main_topics.log 2>&1 &
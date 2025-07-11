#!/bin/bash

mkdir -p outputs/qwen3-8b

python src/qwen.py \
    --dataset_name wiki_titles \
    --dataset_path /aistor/sjtu/hpc_stor01/home/luoyijie/data/wikipedia/wikipedia_wikimedia \
    --template_name wikipedia_titles_wiki_main_topics \
    --topk 3 \
    --output_path outputs/qwen3-8b/wikipedia_titles_wiki_main_topics.jsonl \
    --model_name_or_path /aistor/sjtu/hpc_stor01/home/luoyijie/ckpts/huggingface/Qwen3-8B \
    --data_parallel_size 8 --batch_size 64 > logs/qwen3-8b_wikipedia_titles_wiki_main_topics.log 2>&1 &
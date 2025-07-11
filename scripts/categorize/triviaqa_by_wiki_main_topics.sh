#!/bin/bash

python src/categorize.py \
    --dataset_name triviaqa \
    --dataset_path /aistor/sjtu/hpc_stor01/home/luoyijie/data/wikipedia/triviaqa/raw/unfiltered_nocontext_validation.jsonl \
    --category_file outputs/gpt4o/triviaqa_wiki_main_topics.jsonl \
    --output_dir /aistor/sjtu/hpc_stor01/home/luoyijie/data/wikipedia/triviaqa/categorized_by_wiki_main_topics \
    --domains Entertainment Culture Geography History Humanities Science
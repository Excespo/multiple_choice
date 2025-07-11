#!/bin/bash

python src/categorize.py \
    --dataset_name natural_questions \
    --dataset_path /aistor/sjtu/hpc_stor01/home/luoyijie/data/wikipedia/nq_open/raw/nq_open_validation.jsonl \
    --category_file outputs/gpt4o/nq_open_wiki_main_topics.jsonl \
    --output_dir /aistor/sjtu/hpc_stor01/home/luoyijie/data/wikipedia/nq_open/categorized_by_wiki_main_topics \
    --domains Entertainment Culture Geography History Humanities Science
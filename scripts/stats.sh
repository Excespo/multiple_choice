#!/bin/bash

data_dir=/aistor/sjtu/hpc_stor01/home/luoyijie/src/openai/outputs

# declare -A models=(
#     ["gpt4o"]="${data_dir}/gpt4o/triviaqa_wiki_main_topics.jsonl"
# )

models=(
    "gpt4o"
    # "qwen2.5-1.5b"
    # "qwen2.5-7b"
    # "qwen2.5-14b"
    "qwen3-8b"
)

(
    for model in "${models[@]}"; do
        data_path="${data_dir}/$model/triviaqa_wiki_main_topics.jsonl"
        python src/stats.py --path $data_path
    done
) > logs/stats.log 2>&1 &

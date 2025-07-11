#!/bin/bash

output_dir=/aistor/sjtu/hpc_stor01/home/luoyijie/src/openai/outputs
gpt_output=${output_dir}/gpt4o/triviaqa_wiki_main_topics.jsonl
qwen_output=${output_dir}/qwen3-8b/triviaqa_wiki_main_topics.jsonl

python src/f1.py --gpt-output $gpt_output --qwen-output $qwen_output > logs/f1.log 2>&1 &

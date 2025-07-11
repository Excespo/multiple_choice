#!/bin/bash

KEY="sk-CGMc2J1rEjWy7V5c23A33fF0055e4491963eF71d28B2AaEf" # lyj
# KEY="sk-Rc8kklUxKwikzsTWAcD182Fc384647D3A5141cDa78B691E6" # md
URL="https://api.xi-ai.cn/v1"

python src/openai.py \
    --dataset_name wiki_titles \
    --dataset_path /aistor/sjtu/hpc_stor01/home/luoyijie/data/wikipedia/wikipedia_wikimedia \
    --template_name wikipedia_titles_wiki_main_topics \
    --output_path outputs/gpt4o/wikipedia_titles_wiki_main_topics.jsonl \
    --api_key $KEY --base_url $URL --model gpt-4o --temperature 0.3 --max_tokens 64 \
    --concurrency 1024 --timeout 60 --max_retries 5 > logs/gpt4o_wikipedia_titles_wiki_main_topics.log 2>&1 &

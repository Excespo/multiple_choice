import os
import json
import argparse
import logging

from src.dataset_utils import get_unified_interface, list_available_templates, list_available_datasets
from src.executors.openai_executor import OpenAIExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse():
    parser = argparse.ArgumentParser(description="Process datasets with OpenAI API using the pluggable executor system.")

    # Essential arguments
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of registered dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the final results")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")

    # Model and template arguments
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name to use")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1", help="API base URL")
    parser.add_argument("--template_name", type=str, default=None, help="Template name to use (uses dataset default if not specified)")
    parser.add_argument("--topk", type=int, default=3, help="Top-k value to be passed to the template")

    # API call tuning
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max_retries", type=int, default=3)
    
    # Execution control
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for prompt generation")
    parser.add_argument("--resume_from_raw_results", type=str, default="",
                        help="Path to raw results file to resume post-processing from. Skips API calls.")

    # Listing commands
    parser.add_argument("--list_templates", action="store_true", help="List all available templates and exit")
    parser.add_argument("--list_datasets", action="store_true", help="List all available datasets and exit")

    return parser.parse_args()


async def main(args):
    # Handle list commands
    if args.list_templates:
        templates = list_available_templates()
        print("Available templates:")
        for template_id, description in templates.items():
            print(f"  {template_id}: {description}")
        return
    
    if args.list_datasets:
        datasets = list_available_datasets()
        print("Available datasets:")
        for dataset_name in datasets:
            print(f"  {dataset_name}")
        return

    dataset_interface = get_unified_interface(args.dataset_name, args.template_name)
    logger.info(f"Using dataset: {args.dataset_name} with template: {dataset_interface.template_name}")

    if not args.resume_from_raw_results:
        executor = OpenAIExecutor(
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            timeout=args.timeout,
            max_retries=args.max_retries
        )
        
        processed_results, raw_results = await executor.run(
            dataset_interface, args.dataset_path, args.topk, args.batch_size
        )
        
        raw_results_path = args.output_path.replace('.jsonl', '_raw_results.jsonl')
        logger.info(f"Saving raw results to {raw_results_path}")
        async with aiofiles.open(raw_results_path, 'w', encoding='utf-8') as f:
            for result in raw_results:
                await f.write(json.dumps(result, ensure_ascii=False) + '\n')
    else:
        logger.info(f"Resuming from raw results: {args.resume_from_raw_results}")
        raw_results = []
        async with aiofiles.open(args.resume_from_raw_results, 'r', encoding='utf-8') as f:
            async for line in f:
                if line.strip():
                    raw_results.append(json.loads(line))
        
        # We need to re-run post-processing
        all_data = dataset_interface.load_dataset_shard(args.dataset_path, 0, 1)
        original_questions_map = {entry["id"]: entry["data"] for entry in all_data}
        
        processed = []
        for r in raw_results:
            question = original_questions_map.get(r.get("id"))
            if question is not None and r.get("success"):
                processed.append({"id": r["id"], "question": question, "domains": r.get("domains")})
        processed.sort(key=lambda x: x["id"])
        processed_results = processed

    logger.info(f"Saving final processed results to {args.output_path}")
    async with aiofiles.open(args.output_path, 'w', encoding='utf-8') as f:
        for result in processed_results:
            await f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    logger.info("Processing completed successfully")


if __name__ == "__main__":
    asyncio.run(main(parse())) 
import os
import json
import argparse
import logging

from src.dataset_utils import get_unified_interface, list_available_templates, list_available_datasets
from src.executors.qwen_executor import QwenExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse():
    parser = argparse.ArgumentParser(description="Process datasets with Qwen using the pluggable executor system.")

    parser.add_argument("--dataset_name", type=str, required=True, help="Name of registered dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the final results")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model or its name on Hugging Face Hub")
    
    parser.add_argument("--template_name", type=str, default=None, 
                        help="Template name to use (if not specified, uses dataset default)")
    parser.add_argument("--topk", type=int, default=3, help="Number of top predictions to save")
    
    parser.add_argument("--data_parallel_size", type=int, default=1, help="Number of GPUs to use for data parallelism")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")

    parser.add_argument("--resume_from_raw_results", type=str, default="",
                        help="Path to raw results file to resume post-processing from. Skips model execution.")
    
    parser.add_argument("--list_templates", action="store_true", help="List all available templates and exit")
    parser.add_argument("--list_datasets", action="store_true", help="List all available datasets and exit")

    return parser.parse_args()


def main(args):
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

    # Load unified dataset interface with optional template name
    dataset_interface = get_unified_interface(args.dataset_name, args.template_name)
    
    logger.info(f"Using dataset: {args.dataset_name} with template: {dataset_interface.template_name}")

    if not args.resume_from_raw_results:
        # Initialize and run the executor
        logger.info(f"Initializing QwenExecutor with model: {args.model_name_or_path}")
        executor = QwenExecutor(
            model_path=args.model_name_or_path,
            batch_size=args.batch_size,
            dp_size=args.data_parallel_size
        )

        logger.info(f"Processing dataset using {executor.dp_size} workers...")
        results = executor.run(dataset_interface, args.dataset_path)

        # Save raw results for potential resumption
        raw_results_path = args.output_path.replace('.jsonl', '_raw_results.jsonl')
        logger.info(f"Saving raw results to {raw_results_path} for potential reuse.")
        with open(raw_results_path, "w", encoding="utf-8") as fout:
            for result in results:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    else:
        with open(args.resume_from_raw_results, "rt", encoding="utf-8") as fin:
            results = [json.loads(line) for line in fin]
        logger.info(f"Loaded {len(results)} raw records from {args.resume_from_raw_results}")

    # Post-process results using the dataset-specific method
    logger.info(f"Post-processing results to get top {args.topk} predictions...")
    processed_results = dataset_interface.batch_process_results(results, topk=args.topk)

    # Save final results
    with open(args.output_path, "w", encoding="utf-8") as fout:
        for result in processed_results:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    logger.info(f"Final results saved to {args.output_path}")


if __name__ == "__main__":
    main(parse()) 
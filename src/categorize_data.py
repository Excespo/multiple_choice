import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any

from src.dataset_utils import get_unified_interface, list_available_templates, list_available_datasets

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_category_file(category_file: str) -> Dict[int, Dict[str, Any]]:
    """
    Loads category information from a JSONL file.
    Each line is expected to be a JSON object containing 'id', 'question', and 'domains'.
    Returns a dictionary mapping IDs to their category information.
    """
    id_to_category_info = {}
    
    with open(category_file, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    
                    if "id" not in entry or "question" not in entry or "domains" not in entry:
                        logger.warning(f"Skipping malformed line in category file (missing 'id', 'question', or 'domains'): {line.strip()}")
                        continue
                    
                    entry_id = entry["id"]
                    
                    # The 'question' field can be a string or a nested object.
                    question_data = entry["question"]
                    if isinstance(question_data, dict) and "question" in question_data:
                        question = question_data["question"]
                    elif isinstance(question_data, str):
                        question = question_data
                    else:
                        logger.warning(f"Skipping line with unexpected question format for ID {entry_id}: {line.strip()}")
                        continue
                        
                    if entry_id in id_to_category_info:
                        logger.warning(f"Duplicate ID found in category file: '{entry_id}'. Overwriting entry.")
                    
                    id_to_category_info[entry_id] = {
                        "question": question,
                        "domains": entry["domains"]
                    }
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Skipping line with invalid JSON or type error: {line.strip()}")

    return id_to_category_info

def categorize_dataset(
    dataset_name: str,
    dataset_path: str,
    category_file: str,
    output_dir: str,
    selected_domains: List[str] = None
) -> None:
    """
    Categorize dataset entries by domains and save to separate files.
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load category information, mapping ID to category info
    id_to_category_info = load_category_file(category_file)
    logger.info(f"Loaded category information for {len(id_to_category_info)} unique IDs.")

    # Load the raw dataset directly from file to preserve original format
    logger.info(f"Loading raw dataset from path '{dataset_path}'")
    raw_dataset_entries = []
    try:
        with open(dataset_path, "rt", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if line.strip():
                    try:
                        entry = json.loads(line)
                        # Add an ID based on line number (0-indexed)
                        entry_with_id = {"id": line_idx, "raw_data": entry}
                        raw_dataset_entries.append(entry_with_id)
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line {line_idx + 1}: {line.strip()}")
    except Exception as e:
        logger.error(f"Failed to load raw dataset: {e}")
        return
    
    logger.info(f"Finished loading raw dataset with {len(raw_dataset_entries)} entries.")

    # Prepare a dictionary to hold the verified dataset entries for each domain
    domain_to_final_entries = {}
    question_match_count = 0
    question_mismatch_count = 0
    id_miss_count = 0
    
    # Iterate through each raw dataset entry, find its category by ID, and verify by question
    for entry_with_id in raw_dataset_entries:
        entry_id = entry_with_id["id"]
        raw_entry = entry_with_id["raw_data"]
        
        dataset_question = raw_entry.get("question")

        if entry_id in id_to_category_info:
            cat_info = id_to_category_info[entry_id]
            
            # Verify that the questions are identical
            if cat_info.get("question") == dataset_question:
                question_match_count += 1
                # If they match, add the original raw entry to all its domains
                for domain in cat_info.get("domains", []):
                    # If a domain filter is active, respect it
                    if selected_domains and domain not in selected_domains:
                        continue
                    
                    if domain not in domain_to_final_entries:
                        domain_to_final_entries[domain] = []
                    # Save the original raw entry exactly as it was in the file
                    domain_to_final_entries[domain].append(raw_entry)
            else:
                question_mismatch_count += 1
                logger.debug(
                    f"Question mismatch for ID {entry_id}. "
                    f"Dataset question: '{dataset_question}'. "
                    f"Category question: '{cat_info.get('question')}'."
                )
        else:
            id_miss_count += 1

    logger.info(f"Processed {len(raw_dataset_entries)} entries.")
    logger.info(f"Question matches: {question_match_count}")
    
    if question_mismatch_count > 0:
        logger.warning(f"Found and skipped {question_mismatch_count} entries due to question mismatches.")
    if id_miss_count > 0:
        logger.warning(f"Could not find category info for {id_miss_count} dataset entries (ID miss).")

    # Save the categorized entries to files
    if not domain_to_final_entries:
        logger.warning("No entries to save after filtering and verification.")
        return

    for domain, entries in domain_to_final_entries.items():
        if not entries:
            continue
            
        domain_filename = "".join(c if c.isalnum() else "_" for c in domain)
        domain_file = output_path / f"{domain_filename}.jsonl"
        
        with open(domain_file, "wt", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        logger.info(f"Saved {len(entries)} entries to {domain_file}")

def main():
    parser = argparse.ArgumentParser(description="Categorize dataset entries by domains")
    
    parser.add_argument("--dataset_name", type=str, required=True, 
                        help="Name of the registered dataset to use")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the dataset file")
    parser.add_argument("--category_file", type=str, required=True, 
                        help="Path to the category file with domain information (JSONL format)")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save categorized files")
    parser.add_argument("--domains", nargs="+", type=str, 
                        help="Specific domains to categorize (optional)")
    
    args = parser.parse_args()
    
    logger.info(f"Starting categorization for dataset: {args.dataset_name}")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Using category file: {args.category_file}")
    if args.domains:
        logger.info(f"Filtering for domains: {args.domains}")
    
    categorize_dataset(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        category_file=args.category_file,
        output_dir=args.output_dir,
        selected_domains=args.domains
    )
    
    logger.info("Categorization complete!")

if __name__ == "__main__":
    main() 
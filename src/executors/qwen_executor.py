import torch.multiprocessing as mp
from typing import Dict, Any, List
import logging
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..utils import load_pretrained
from ..dataset_utils import UnifiedDatasetInterface

logger = logging.getLogger(__name__)

def _worker_main(
    rank: int,
    world_size: int,
    model_path: str,
    batch_size: int,
    dataset_interface: UnifiedDatasetInterface,
    dataset_path: str,
    results_queue: mp.Queue
):
    """
    Worker process for data parallel evaluation.
    
    Each worker loads the model onto a specific device (NPU/GPU) and processes
    a unique shard of the dataset.
    """
    # Setup device based on rank. Assumes torch_npu for 'npu' devices.
    device = torch.device(f"npu:{rank}")
    os.environ["RANK"] = str(rank)
    
    try:
        # Load model and tokenizer for this worker
        model, tokenizer = load_pretrained(model_path, torch_dtype="auto")
        model = model.to(device)
        model.eval()

        # Load the specific data shard for this worker
        data_shard = dataset_interface.load_dataset_shard(dataset_path, shard_index=rank, num_shards=world_size)
        choices = dataset_interface.choices
        
        if rank == 0:
            logger.info(f"Worker {rank} processing {len(data_shard)} entries with {len(choices)} choices.")

        # Process the shard in batches
        for i in range(0, len(data_shard), batch_size):
            batch_data = data_shard[i:i + batch_size]
            batch_results = []

            for entry in batch_data:
                question = entry["data"]
                # Use the dataset-specific template to generate the final prompt
                prompt = dataset_interface.apply_template_to_data(question, topk=1)
                
                log_likelihoods = {}
                
                # For a single question, create a batch of inputs where each input
                # is the prompt concatenated with one of the possible choices.
                prompts_and_choices = [prompt + choice for choice in choices]
                inputs = tokenizer(prompts_and_choices, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
                
                # We need the length of the prompt tokens to isolate the choice tokens later
                prompt_tokens_count = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]
                
                with torch.no_grad():
                    outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
                    logits = outputs.logits

                # Calculate log-likelihood for each choice in the batch
                for choice_idx, choice in enumerate(choices):
                    # The choice tokens in the combined input
                    choice_ids = tokenizer(choice, add_special_tokens=False).input_ids
                    
                    # Logits are shifted by one position relative to input_ids for next-token prediction.
                    # The logits for the choice tokens start right after the prompt ends.
                    start_index = prompt_tokens_count
                    end_index = start_index + len(choice_ids)
                    
                    # Get the specific logits corresponding to the positions of the choice tokens
                    choice_logits = logits[choice_idx, start_index-1:end_index-1, :]
                    
                    # Calculate the log probabilities from the logits
                    log_probs = F.log_softmax(choice_logits, dim=-1)
                    
                    # Create a tensor of the choice token IDs to use for gathering
                    choice_token_ids_tensor = torch.tensor(choice_ids, device=device).unsqueeze(-1)
                    
                    # Gather the log probabilities of the actual choice tokens and sum them up
                    gathered_log_probs = torch.gather(log_probs, 1, choice_token_ids_tensor).sum()
                    
                    log_likelihoods[choice] = gathered_log_probs.item()

                batch_results.append({
                    "id": entry["id"],
                    "question": question,
                    "loglikelihoods": log_likelihoods
                })
            
            # Send the results for the processed batch back to the main process
            results_queue.put(batch_results)

    except Exception as e:
        logger.error(f"Error in worker {rank}: {e}", exc_info=True)
        results_queue.put(f"ERROR_IN_WORKER_{rank}") # Signal error
    finally:
        results_queue.put(None) # Signal completion for this worker


class QwenExecutor:
    """
    Executor for running evaluations on Qwen-like models using data parallelism.
    """
    def __init__(self, model_path: str, batch_size: int, dp_size: int):
        if dp_size > torch.npu.device_count():
            raise ValueError(f"Requested {dp_size} devices, but only {torch.npu.device_count()} are available.")
        
        self.model_path = model_path
        self.batch_size = batch_size
        self.dp_size = dp_size

    @property
    def world_size(self):
        return self.dp_size

    def run(self, dataset_interface: UnifiedDatasetInterface, dataset_path: str) -> List[Dict[str, Any]]:
        """
        Runs the evaluation in parallel across multiple devices.
        
        It spawns a worker process for each device, which handles a fraction of
        the data. The main process collects results from all workers.
        """
        ctx = mp.get_context("spawn")
        results_queue = ctx.Queue()

        # Get total dataset size for the progress bar
        total_size = dataset_interface.get_dataset_info(dataset_path)["size"]
        
        args = (
            self.world_size,
            self.model_path,
            self.batch_size,
            dataset_interface,
            dataset_path,
            results_queue
        )

        logger.info(f"Spawning {self.world_size} worker processes...")
        # `join=False` allows the main process to collect results while workers run
        mp.spawn(_worker_main, args=args, nprocs=self.world_size, join=False)

        all_results = []
        completed_workers = 0
        with tqdm(total=total_size, desc="Evaluating with QwenExecutor") as pbar:
            while completed_workers < self.world_size:
                # Block and wait for a result from any worker
                result = results_queue.get()
                
                if result is None:
                    # A `None` from the queue signals one worker has finished
                    completed_workers += 1
                elif isinstance(result, str) and result.startswith("ERROR_IN_WORKER"):
                    logger.error(f"Received error signal from a worker: {result}")
                    completed_workers += 1 # A failed worker is also a completed one
                else:
                    # This is a list of processed entries
                    all_results.extend(result)
                    pbar.update(len(result))
        
        # Sort results by their original ID to maintain dataset order
        all_results.sort(key=lambda x: x["id"])
        
        logger.info(f"Finished evaluation. Collected {len(all_results)} results.")
        return all_results 
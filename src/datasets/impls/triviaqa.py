import json
import sys
import os
from typing import Iterator, Dict, Any, List

# Add current directory to Python path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from ..base import BaseDataset, DatasetRegistry


class TriviaQADataset(BaseDataset):
    """TriviaQA dataset implementation focused on data processing."""
    
    @property
    def dataset_name(self) -> str:
        return "triviaqa"
    
    def load_dataset(self, path: str, shard_index: int = 0, num_shards: int = 1) -> List[Dict[str, Any]]:
        """Load TriviaQA dataset from JSONL file and return data for specific shard."""
        try:
            all_data = []
            with open(path, "rt", encoding="utf-8") as fin:
                for i, line in enumerate(fin):
                    entry = json.loads(line)
                    all_data.append({"id": i, "data": entry["question"]})
            
            print(f"TriviaQA dataset loaded with {len(all_data)} entries")
            
            # Return specific shard for data parallel processing
            if num_shards > 1:
                shards = self.create_data_shards(all_data, num_shards)
                return shards[shard_index]
            else:
                return all_data
                
        except Exception as e:
            raise RuntimeError(f"Failed to load TriviaQA dataset from {path}: {e}")
    
    def get_dataset_size(self, path: str) -> int:
        """Get dataset size efficiently by counting lines."""
        try:
            with open(path, "rt", encoding="utf-8") as fin:
                return sum(1 for _ in fin)
        except Exception as e:
            raise RuntimeError(f"Failed to get TriviaQA dataset size from {path}: {e}")
    
    def apply_template(self, entry_data: str, template_provider) -> str:
        """Apply template to TriviaQA data."""
        # For TriviaQA, we just pass the question as is
        return template_provider.render(entry_data)
    
    def batch_apply_templates(self, entries: List[Dict[str, Any]], template_provider, topk: int = 3) -> List[str]:
        """Efficiently apply templates in batch."""
        return [template_provider.render(entry["data"], topk=topk) for entry in entries]
    
    def post_process_results(self, results: List[Dict[str, Any]], topk: int = 3) -> List[Dict[str, Any]]:
        """Process model results and extract top-k predictions."""
        processed = []
        
        for i, result in enumerate(results):
            # Extract question text
            q = result["question"]
            q = q.split("[Q]: ")[-1]
            q = q.split("[Topic]")[0]
            q = q.strip()
            
            # Sort loglikelihoods and get top-k
            llh = result["loglikelihoods"]
            llh = sorted(llh.items(), key=lambda x: x[1], reverse=True)[:topk]
            
            top_keys = [k for k, _ in llh]
            top_nlls = [v for _, v in llh]
            
            processed.append({
                "id": i,
                "question": q,
                "domains": top_keys,
                "loglikelihoods": top_nlls,
                "records": llh
            })
            
        return processed


# Register the dataset
DatasetRegistry.register("triviaqa", TriviaQADataset) 
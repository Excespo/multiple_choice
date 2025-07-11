import os
import sys
import datasets
from typing import Iterator, Dict, Any, List

# Add current directory to Python path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from ..base import BaseDataset, DatasetRegistry


class WikiTitlesDataset(BaseDataset):
    """Wikipedia titles dataset implementation focused on data processing."""
    
    @property
    def dataset_name(self) -> str:
        return "wiki_titles"
    
    def load_dataset(self, path: str, shard_index: int = 0, num_shards: int = 1) -> List[Dict[str, Any]]:
        """Load Wikipedia titles dataset and return data for specific shard."""
        try:
            dataset = datasets.load_dataset(path, num_proc=os.cpu_count())["train"]
            print(f"Dataset loaded with {len(dataset)} entries")
            
            # Convert to list format
            all_data = [{"id": i, "data": entry["title"]} for i, entry in enumerate(dataset)]
            
            # Return specific shard for data parallel processing
            if num_shards > 1:
                shards = self.create_data_shards(all_data, num_shards)
                return shards[shard_index]
            else:
                return all_data
                
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from {path}: {e}")
    
    def get_dataset_size(self, path: str) -> int:
        """Get dataset size efficiently without loading all data."""
        try:
            dataset = datasets.load_dataset(path, num_proc=os.cpu_count())["train"]
            return len(dataset)
        except Exception as e:
            raise RuntimeError(f"Failed to get dataset size from {path}: {e}")
    
    def apply_template(self, entry_data: str, template_provider) -> str:
        """Apply template to Wikipedia title data."""
        # For Wikipedia titles, we just pass the title as the question
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
DatasetRegistry.register("wiki_titles", WikiTitlesDataset) 
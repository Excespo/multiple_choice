from typing import Dict, Any, List
import importlib
import os
import sys

# Add current directory to Python path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from .base import DatasetRegistry, BaseDataset


class DatasetManager:
    """
    Manager for loading and using datasets through a unified interface.
    """
    
    def __init__(self):
        """Initialize the dataset manager and auto-discover datasets."""
        self._auto_discover_datasets()
    
    def _auto_discover_datasets(self):
        """Automatically discover and import all dataset implementations from the 'impls' directory."""
        package_dir = os.path.dirname(__file__)
        impls_dir = os.path.join(package_dir, 'impls')
        
        for filename in os.listdir(impls_dir):
            if filename.endswith('.py') and not filename.startswith('__'):
                module_name = f".impls.{filename[:-3]}"
                try:
                    importlib.import_module(module_name, package=__package__)
                except ImportError as e:
                    print(f"Warning: Could not import dataset implementation {module_name}: {e}")
    
    def list_available_datasets(self) -> List[str]:
        """List all available datasets."""
        return DatasetRegistry.list_datasets()
    
    def get_dataset(self, name: str, config: Dict[str, Any] = None) -> BaseDataset:
        """Get a dataset instance by name."""
        return DatasetRegistry.get_dataset(name, config)
    
    def load_dataset_shard(self, dataset_name: str, data_path: str, 
                          shard_index: int = 0, num_shards: int = 1, 
                          config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Load a specific shard of the dataset for data parallel processing.
        
        Args:
            dataset_name: Name of the dataset to use
            data_path: Path to the dataset files
            shard_index: Index of current shard (for data parallel)
            num_shards: Total number of shards (for data parallel)
            config: Dataset-specific configuration
            
        Returns:
            List of entries for the specified shard
        """
        dataset = self.get_dataset(dataset_name, config)
        return dataset.load_dataset(data_path, shard_index, num_shards)
    
    def batch_process_dataset(self, dataset_name: str, data_path: str, 
                             template: str, batch_size: int = 32,
                             shard_index: int = 0, num_shards: int = 1,
                             config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Process a dataset in batch mode for efficiency.
        
        Args:
            dataset_name: Name of the dataset to use
            data_path: Path to the dataset files
            template: Prompt template
            batch_size: Size of processing batches
            shard_index: Index of current shard (for data parallel)
            num_shards: Total number of shards (for data parallel)
            config: Dataset-specific configuration
            
        Returns:
            List of processed entries with prompts
        """
        dataset = self.get_dataset(dataset_name, config)
        entries = dataset.load_dataset(data_path, shard_index, num_shards)
        return dataset.process_batch(entries, template, batch_size)
    
    def batch_process_results(self, dataset_name: str, results: List[Dict[str, Any]], 
                            topk: int = 3, config: Dict[str, Any] = None):
        """
        Process model results using dataset-specific post-processing.
        
        Args:
            dataset_name: Name of the dataset
            results: Raw model results
            topk: Number of top results to keep
            config: Dataset-specific configuration
            
        Returns:
            Processed results
        """
        dataset = self.get_dataset(dataset_name, config)
        return dataset.post_process_results(results, topk)
    
    def get_dataset_info(self, dataset_name: str, data_path: str, 
                        config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get information about a dataset without loading all data.
        
        Args:
            dataset_name: Name of the dataset
            data_path: Path to the dataset files
            config: Dataset-specific configuration
            
        Returns:
            Dictionary with dataset information
        """
        dataset = self.get_dataset(dataset_name, config)
        
        try:
            size = dataset.get_dataset_size(data_path)
            return {
                "name": dataset_name,
                "size": size,
                "dataset_class": dataset.__class__.__name__,
                "config": dataset.config
            }
        except Exception as e:
            return {
                "name": dataset_name,
                "size": -1,
                "error": str(e),
                "dataset_class": dataset.__class__.__name__,
                "config": dataset.config
            }


# Example usage functions
def example_batch_processing():
    """Demonstrate batch processing capabilities."""
    manager = DatasetManager()
    
    print("Available datasets:", manager.list_available_datasets())
    
    # Example of batch processing
    dataset_name = "wiki_titles"
    template = "Question: <QUESTION>\nAnswer:"
    data_path = "path/to/wiki_titles/data"
    
    try:
        # Get dataset info first
        info = manager.get_dataset_info(dataset_name, data_path)
        print(f"\nDataset info: {info}")
        
        # Process in batch mode
        processed_entries = manager.batch_process_dataset(
            dataset_name=dataset_name,
            data_path=data_path,
            template=template,
            batch_size=64,  # Process 64 at a time
            shard_index=0,  # First shard
            num_shards=1    # Single process
        )
        
        print(f"Processed {len(processed_entries)} entries in batch")
        
        # Show first few examples
        for entry in processed_entries[:3]:
            print(f"ID: {entry['id']}, Prompt: {entry['prompt'][:50]}...")
            
    except Exception as e:
        print(f"Error in batch processing: {e}")


def example_data_parallel_processing():
    """Demonstrate data parallel processing simulation."""
    manager = DatasetManager()
    
    dataset_name = "wiki_titles"
    template = "Question: <QUESTION>\nAnswer:"
    data_path = "path/to/wiki_titles/data"
    num_workers = 4  # Simulate 4 parallel workers
    
    print(f"\nSimulating data parallel processing with {num_workers} workers:")
    
    try:
        # Get total dataset info
        info = manager.get_dataset_info(dataset_name, data_path)
        print(f"Total dataset size: {info.get('size', 'unknown')}")
        
        # Simulate parallel processing across workers
        all_results = []
        for worker_id in range(num_workers):
            print(f"\nWorker {worker_id} processing shard {worker_id}/{num_workers}")
            
            # Each worker processes its shard
            worker_results = manager.batch_process_dataset(
                dataset_name=dataset_name,
                data_path=data_path,
                template=template,
                batch_size=32,
                shard_index=worker_id,
                num_shards=num_workers
            )
            
            print(f"Worker {worker_id} processed {len(worker_results)} entries")
            all_results.extend(worker_results)
        
        print(f"\nTotal processed entries across all workers: {len(all_results)}")
        
    except Exception as e:
        print(f"Error in data parallel processing: {e}")


def example_efficient_loading():
    """Demonstrate efficient data loading patterns."""
    manager = DatasetManager()
    
    dataset_name = "wiki_titles"
    data_path = "path/to/wiki_titles/data"
    
    try:
        # Method 1: Load specific shard directly
        print("Loading shard 0 of 4...")
        shard_data = manager.load_dataset_shard(
            dataset_name, data_path, 
            shard_index=0, num_shards=4
        )
        print(f"Loaded {len(shard_data)} entries for shard 0")
        
        # Method 2: Get dataset info without loading data
        info = manager.get_dataset_info(dataset_name, data_path)
        print(f"Dataset info (no data loaded): {info}")
        
    except Exception as e:
        print(f"Error in efficient loading: {e}")


if __name__ == "__main__":
    print("=== Batch Processing Example ===")
    example_batch_processing()
    
    print("\n=== Data Parallel Processing Example ===")
    example_data_parallel_processing()
    
    print("\n=== Efficient Loading Example ===")
    example_efficient_loading() 
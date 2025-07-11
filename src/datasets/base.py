from abc import ABC, abstractmethod
from typing import Dict, Any, List

# Global registry for datasets
_dataset_registry = {}


class BaseDataset(ABC):
    """
    Abstract base class for all dataset implementations.
    Provides a unified interface for dataset processing. 
    Template management is now handled separately by TemplateProvider classes.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the dataset with configuration.
        
        Args:
            config: Dataset-specific configuration parameters
        """
        self.config = config or {}
    
    @abstractmethod
    def load_dataset(self, path: str, shard_index: int = 0, num_shards: int = 1) -> List[Dict[str, Any]]:
        """
        Load dataset and return all data as a list for batch processing.
        
        Args:
            path: Path to the dataset
            shard_index: Index of current shard (for data parallel)
            num_shards: Total number of shards (for data parallel)
            
        Returns:
            List of dicts with 'id' and 'data' keys
        """
        pass
    
    def apply_template(self, entry_data: str, template_provider) -> str:
        """
        Apply template to entry data using the provided template provider.
        
        Args:
            entry_data: Raw data from dataset entry
            template_provider: TemplateProvider instance
            
        Returns:
            Formatted prompt string
        """
        # Default implementation just passes the data to template provider
        # Subclasses can override for dataset-specific processing
        return template_provider.render(entry_data)
    
    def batch_apply_templates(self, entries: List[Dict[str, Any]], template_provider, topk: int = 3) -> List[str]:
        """
        Apply templates to multiple entries in batch for better efficiency.
        
        Args:
            entries: List of dataset entries
            template_provider: TemplateProvider instance
            topk: Number of top predictions to request
            
        Returns:
            List of formatted prompt strings
        """
        return [template_provider.render(entry["data"], topk=topk) for entry in entries]
    
    @abstractmethod
    def post_process_results(self, results: List[Dict[str, Any]], topk: int = 3) -> List[Dict[str, Any]]:
        """
        Post-process model results.
        
        Args:
            results: Raw results from model
            topk: Number of top results to keep
            
        Returns:
            Processed results
        """
        pass
    
    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Return the name of this dataset."""
        pass
    
    @property
    def default_template_name(self) -> str:
        """Return the default template name for this dataset."""
        # Default to wiki_main_topics, can be overridden by subclasses
        return "wiki_main_topics"
    
    def validate_config(self) -> bool:
        """Validate dataset configuration."""
        return True
    
    def get_dataset_size(self, path: str) -> int:
        """
        Get total size of the dataset without loading all data.
        Should be implemented by subclasses for efficiency.
        
        Args:
            path: Path to the dataset
            
        Returns:
            Total number of entries in the dataset
        """
        # Default implementation loads all data (inefficient)
        return len(self.load_dataset(path))
    
    def create_data_shards(self, data: List[Dict[str, Any]], num_shards: int) -> List[List[Dict[str, Any]]]:
        """
        Split data into shards for data parallel processing.
        
        Args:
            data: Full dataset
            num_shards: Number of shards to create
            
        Returns:
            List of data shards
        """
        shard_size = len(data) // num_shards
        remainder = len(data) % num_shards
        
        shards = []
        start_idx = 0
        
        for i in range(num_shards):
            # Distribute remainder across first few shards
            current_shard_size = shard_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_shard_size
            shards.append(data[start_idx:end_idx])
            start_idx = end_idx
            
        return shards
    
    def process_batch(self, entries: List[Dict[str, Any]], template_provider, 
                     batch_size: int = 32, topk: int = 3) -> List[Dict[str, Any]]:
        """
        Process a batch of entries with batched prompt generation.
        
        Args:
            entries: List of dataset entries
            template_provider: TemplateProvider instance
            batch_size: Size of processing batches
            topk: Number of top predictions to request
            
        Returns:
            List of processed entries with prompts
        """
        processed = []
        
        for i in range(0, len(entries), batch_size):
            batch = entries[i:i + batch_size]
            prompts = self.batch_apply_templates(batch, template_provider, topk)
            
            for entry, prompt in zip(batch, prompts):
                processed.append({
                    "id": entry["id"],
                    "raw_data": entry["data"],
                    "prompt": prompt,
                    "dataset": self.dataset_name
                })
                
        return processed


class DatasetRegistry:
    """Registry for managing dataset implementations."""
    
    _datasets = {}
    
    @classmethod
    def register(cls, name: str, dataset_class: type):
        """Register a dataset implementation."""
        if not issubclass(dataset_class, BaseDataset):
            raise ValueError(f"Dataset class must inherit from BaseDataset")
        cls._datasets[name] = dataset_class
    
    @classmethod
    def get_dataset(cls, name: str, config: Dict[str, Any] = None) -> BaseDataset:
        """Get a dataset instance by name."""
        if name not in cls._datasets:
            raise ValueError(f"Unknown dataset: {name}")
        return cls._datasets[name](config)
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all registered datasets."""
        return list(cls._datasets.keys()) 
"""
Unified dataset interface utilities.
Provides simplified access to datasets with pluggable template management.
"""

from typing import Tuple, Optional

from .datasets.manager import DatasetManager
from .templates import TemplateRegistry

def get_unified_interface(dataset_name: str, template_name: Optional[str] = None) -> 'UnifiedDatasetInterface':
    """
    Get unified dataset interface with optional template specification.
    
    Args:
        dataset_name: Name of the dataset
        template_name: Optional template name (uses dataset default if not specified)
        
    Returns:
        UnifiedDatasetInterface instance with all functionality
    """
    return UnifiedDatasetInterface(dataset_name, template_name)


def list_available_templates() -> dict:
    """List all available templates with descriptions."""
    return TemplateRegistry.get_template_info()


def list_available_datasets() -> list:
    """List all available datasets."""
    manager = DatasetManager()
    return manager.list_available_datasets()


class UnifiedDatasetInterface:
    """
    Unified interface that provides all functionality with pluggable template support.
    """
    
    def __init__(self, dataset_name: str, template_name: Optional[str] = None):
        self.dataset_name = dataset_name
        self.manager = DatasetManager()
        self.dataset = self.manager.get_dataset(dataset_name)
        
        # Get template provider
        if template_name is None:
            template_name = self.dataset.default_template_name
        
        self.template_name = template_name
        self.template_provider = TemplateRegistry.get_template(template_name)
    
    def get_template_with_topk(self, topk: int = 3) -> str:
        """Get template configured for topk predictions."""
        return self.template_provider.render("", topk=topk)
    
    @property
    def template(self) -> str:
        """Get default template."""
        return self.template_provider.render("", topk=1)
    
    @property
    def choices(self) -> list:
        """Get valid choices."""
        return self.template_provider.choices
    
    def get_dataset_info(self, data_path: str):
        """Get dataset information."""
        return self.manager.get_dataset_info(self.dataset_name, data_path)
    
    def load_dataset_shard(self, data_path: str, shard_index: int = 0, num_shards: int = 1):
        """Load dataset shard."""
        return self.manager.load_dataset_shard(
            self.dataset_name, data_path, shard_index, num_shards
        )
    
    def batch_process_dataset(self, data_path: str, topk: int = 3, batch_size: int = 32,
                             shard_index: int = 0, num_shards: int = 1):
        """Process dataset in batch mode using template provider."""
        # Load the dataset shard
        entries = self.load_dataset_shard(data_path, shard_index, num_shards)
        
        # Process using template provider
        return self.dataset.process_batch(entries, self.template_provider, batch_size, topk)
    
    def batch_process_results(self, results: list, topk: int = 3):
        """Process model results."""
        return self.manager.batch_process_results(self.dataset_name, results, topk)
    
    def apply_template_to_data(self, data: str, topk: int = 3) -> str:
        """Apply template to a single piece of data."""
        return self.dataset.apply_template(data, self.template_provider)
    
    def render_template(self, question: str, topk: int = 3) -> str:
        """Directly render template with question."""
        return self.template_provider.render(question, topk=topk)
    
    @property
    def available_templates(self) -> list:
        """Get list of available template names."""
        return TemplateRegistry.list_templates()
    
    def switch_template(self, template_name: str):
        """Switch to a different template."""
        self.template_name = template_name
        self.template_provider = TemplateRegistry.get_template(template_name) 
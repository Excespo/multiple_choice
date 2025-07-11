"""
Template management system for flexible prompt generation.
Provides pluggable TemplateProvider interface for different prompting strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import importlib
import os


class TemplateProvider(ABC):
    """
    Abstract base class for template providers.
    Allows flexible prompt generation strategies that can be shared across datasets.
    """
    
    @property
    @abstractmethod
    def template_id(self) -> str:
        """Unique identifier for this template."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of this template."""
        pass
    
    @abstractmethod
    def render(self, question: str, *, topk: int = 3) -> str:
        """
        Render the template with the given question.
        
        Args:
            question: The input question/text
            topk: Number of top predictions to request
            
        Returns:
            Formatted prompt string
        """
        pass
    
    @property
    @abstractmethod
    def choices(self) -> List[str]:
        """Get the list of valid choices for this template."""
        pass
    
    @property
    def default_template(self) -> str:
        """Get the default (single choice) template."""
        return self.render("", topk=1)
    
    def validate_config(self) -> bool:
        """Validate template configuration."""
        return True


class TemplateRegistry:
    """Registry for managing template providers."""
    
    _templates = {}
    
    @classmethod
    def register(cls, template_provider: TemplateProvider):
        """Register a template provider."""
        if not isinstance(template_provider, TemplateProvider):
            raise ValueError("Template must inherit from TemplateProvider")
        cls._templates[template_provider.template_id] = template_provider
    
    @classmethod
    def get_template(cls, template_id: str) -> TemplateProvider:
        """Get a template provider by ID."""
        if template_id not in cls._templates:
            raise ValueError(f"Unknown template: {template_id}. Available templates: {cls.list_templates()}")
        return cls._templates[template_id]
    
    @classmethod
    def list_templates(cls) -> List[str]:
        """List all registered template IDs."""
        return list(cls._templates.keys())
    
    @classmethod
    def get_template_info(cls) -> Dict[str, str]:
        """Get information about all registered templates."""
        return {
            template_id: provider.description 
            for template_id, provider in cls._templates.items()
        }


def auto_discover_templates():
    """Automatically discover and import all template implementations."""
    current_dir = os.path.dirname(__file__)
    
    for filename in os.listdir(current_dir):
        if (filename.endswith('.py') and 
            not filename.startswith('__') and 
            filename != 'base.py'):
            
            module_name = filename[:-3]  # Remove .py extension
            try:
                importlib.import_module(f'.{module_name}', package=__package__)
            except ImportError as e:
                print(f"Warning: Could not import template module {module_name}: {e}")


# Auto-discover templates when module is imported
auto_discover_templates()

# Export main classes
__all__ = ['TemplateProvider', 'TemplateRegistry'] 
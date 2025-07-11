import sys
import importlib.util
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

def load_pretrained(model_name_or_path, torch_dtype="auto", **kwargs):
    """
    Load a pretrained model and tokenizer, with support for trusting remote code.
    
    Args:
        model_name_or_path (str): The path to the model or its name on Hugging Face Hub.
        torch_dtype (str or torch.dtype): The desired dtype for the model.
        **kwargs: Additional keyword arguments to pass to the from_pretrained methods.
        
    Returns:
        tuple: A tuple containing the model and the tokenizer.
    """
    logger.info(f"Loading model from: {model_name_or_path}")
    
    # Merge the essential trust_remote_code=True with any other kwargs.
    model_kwargs = {"trust_remote_code": True, **kwargs}
    tokenizer_kwargs = {"trust_remote_code": True, **kwargs}

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            **tokenizer_kwargs
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            **model_kwargs
        )
        
        # Ensure pad_token is set for tokenizers that need it
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Tokenizer `pad_token` was not set, using `eos_token` as default.")
            
        logger.info("Model and tokenizer loaded successfully.")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Failed to load model or tokenizer from {model_name_or_path}: {e}")
        raise


# Legacy function - no longer used with new TemplateProvider system
# def load_module_functions(module_path, module_name, functions):
#     spec = importlib.util.spec_from_file_location(module_name, module_path)
#     module = importlib.util.module_from_spec(spec)
#     sys.modules[module_name] = module
#     spec.loader.exec_module(module)
# 
#     return (
#         getattr(module, fn, None) 
#         for fn in functions
#     )
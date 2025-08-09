
from typing import Dict, Optional
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import find_format_keys


class BatchLayout(Enum):
    """Batch Layout Enum"""
    CONCAT = "concat"
    STACK = "stack"


class PaddingSide(Enum):
    """Padding Side Enum"""
    LEFT = "left"
    RIGHT = "right"


class TruncateSide(Enum):
    """Truncate Side Enum"""
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


class PromptType(Enum):
    """Prompt Types Enum"""
    
    SYSTEM = "system"
    CONTEXT = "context"
    QUERY = "query"
    RESPONSE = "response"
    PROMPT = "prompt" # NOTE: prompt = system + context + query
    ALL = "all" # NOTE: all = prompt + response
    

class PromptTemplate(nn.Module):
    """Prompt Template module"""
    
    def __init__(self, template_str: str = ""):
        """Initialize Prompt Template module
        
        Args:
            template_str (str): the template string with the format: "....{key1}...{key2}..."
        """
        super().__init__()
        self.template_str = template_str
        self.default_dict = {}
        
         
    def keys(self) -> Dict[str, Optional[str]]:
        """Get the keys with its default values of the prompt template as a dictionary
        NOTE: if any key has not been set with default value, then use `None` as a placeholder
        """
        if len(self.default_dict) == 0:
            template_keys = find_format_keys(self.template_str)
            for key in template_keys:
                self.default_dict[key] = None
        return self.default_dict
            
         
    def set_default(self, **kwargs: Optional[Dict[str, str]]) -> None:
        """Set the default values of the prompt template keys
        NOTE: ignore the keys in kwargs but not in the template string
        """
        template_keys = find_format_keys(self.template_str)
        for key in template_keys:
            if key not in kwargs:
                self.default_dict[key] = None
            else:
                self.default_dict[key] = kwargs[key]
    
    def forward(self, **kwargs: Optional[Dict[str, str]]) -> str:
        """Set the prompt template keys with the given keyword argument to get the formatted prompt
        NOTE:
            1. if certain prompt template key has not been set with its default value, then its corresponding kwarg should be provided
            2. if certain key in the kwargs is not found in the keys of the prompt template, just ignore it
        """
        keys = self.keys()
        template_dict = {**self.default_dict}
        for key in kwargs:
            if key in keys:
                template_dict[key] = kwargs[key]
        for item in template_dict.items():
            if item[1] is None:
                raise ValueError(f"Prompt template key {item[0]} has not been set with a valid value")
        return self.template_str.format(**template_dict)

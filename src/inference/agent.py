from typing import List, Dict, Tuple, Optional, Union
from contextlib import contextmanager
from enum import Enum
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modeling.models.base import BaseTokenizer, BaseModel

from ..modeling.datasets.base import BatchLayout, PaddingSide, TruncateSide

from ..modeling.prompt import PromptType, PromptTemplate

from ..modeling.config import (
    BaseConfig,
    config_dataclass,
    make_required_field,
    make_fixed_field,
)

from ..utils import convert_to_list


class DecodeStrategy(Enum):
    """Decode Strategies Enum"""
    
    GREEDY = "greedy"
    SAMPLING = "sampling"


@config_dataclass
class InferenceConfig(BaseConfig):
    """Inference Configurations Dataclass"""
    
    # generation configurations
    decode_strategy: DecodeStrategy = DecodeStrategy.GREEDY
    temperature: float = 1.0
    max_new_tokens: int = make_required_field() # NOTE: we allow neither infinite generation nor early stopping for simplicity
    top_p: float = 1.0 # NOTE: only used when using sampling decode strategy
    top_k: int = 50 # NOTE: only used when using sampling decode strategy
    streaming: bool = False # NOTE: used when only one single user query is requested at a time, i.e. `inferred_batch_size == 1`
    sampling_seed: Optional[int] = None # NOTE: only used when using sampling decode strategy, if None then do not set seed
    
    # padding configurations
    batch_layout: BatchLayout = make_fixed_field(BatchLayout.STACK) # NOTE: we only allow stacking for simplicity
    padding_side: PaddingSide = PaddingSide.LEFT
    pad_to_multiple_of: int = 1
    
    # truncate configurations
    truncate_length: Optional[int] = None # NOTE: if None, then no truncation
    truncate_side: TruncateSide = TruncateSide.RIGHT
    
    # common configurations
    device: str = "cpu"

    def __post_init__(self):
        """Post-initialization method for InferenceConfig"""
        super().__post_init__()

        assert self.pad_to_multiple_of > 0 and (
            (self.pad_to_multiple_of & (self.pad_to_multiple_of - 1)) == 0
        ), "pad_to_multiple_of must be a power of 2"

        if self.truncate_length is not None and self.truncate_side == TruncateSide.MIDDLE:
            assert self.truncate_length % 2 == 0, "truncate_length must be even when truncate_side is MIDDLE"


class InferenceAgent(nn.Module):
    """Inference Agent module"""
    
    def __init__(
        self,
        config: InferenceConfig,
        model: BaseModel,
        tokenizer: BaseTokenizer,
    ):
        """Initialize Inference Agent module
        
        Args:
            config (InferenceConfig): Inference Configurations
            model (BaseModel): the inner causal language model, which supports the common APIs of `BaseModel`
            tokenizer (BaseTokenizer): the inner tokenizer, which supports the common APIs of `BaseTokenizer`
        """
        super().__init__()
        self.config = config
        self.model = model.eval().to(config.device)
        self.tokenizer = tokenizer
        
        self._system_prompt = PromptTemplate()
        self._context_prompt = PromptTemplate()
        
    def set_prompt(
        self,
        prompt_template: PromptTemplate,
        prompt_type: PromptType = PromptType.SYSTEM,
    ) -> None:
        """Set the prompt template
        
        Args:
            prompt_template (PromptTemplate): the prompt template
            prompt_type (PromptType): the prompt type
        """
        if prompt_type == PromptType.SYSTEM:
            self._system_prompt = prompt_template
        elif prompt_type == PromptType.CONTEXT:
            self._context_prompt = prompt_template
        else:
            raise ValueError(f"Invalid prompt type: {prompt_type}, prompt_type should be either SYSTEM or CONTEXT")
            
    def get_prompt(
        self,
        prompt_type: PromptType = PromptType.SYSTEM
    ) -> PromptTemplate:
        """Get the prompt template
        
        Args:
            prompt_type (PromptType): the prompt type
        
        Returns:
            PromptTemplate: the prompt template
        """
        if prompt_type == PromptType.SYSTEM:
            return self._system_prompt
        elif prompt_type == PromptType.CONTEXT:
            return self._context_prompt
        else:
            raise ValueError(f"Invalid prompt type: {prompt_type}, prompt_type should be either SYSTEM or CONTEXT")
    
    @torch.no_grad()
    def forward(
        self, 
        query: Union[str, List[str]], 
        **kwargs: Optional[Dict[str, str]]
    ) -> List[Dict[PromptType, str]]:
        """The forward pass of the Inference Agent module
        
        Args:
            query (Union[str, List[str]]): a single query prompt or a batch of user query prompts \
                as the core distinct instructions to ask the model to respond, \
                appended to the end of the complete prompt with the same system prompt and context prompt
                NOTE: when is a streaming mode, the query should be a single prompt
            kwargs (dict): additional keyword arguments to be passed to format the prefixed prompt templates
                NOTE: if certain key in `kwargs` are found in both system prompt template and context prompt template, \
                    the corresponding value will share in both of them as well
        Returns:
            List[Dict[PromptType, str]]: the list of dictionaries, \
                each of which should contain every prompt type in `PromptType` (key) and the corresponding prompt (value)
            NOTE: to simplify, we do not use early stopping strategy since the stopping point for each response might vary, \
                thus the length of the latent token ids for each response is ensured to be `max_new_tokens`
        """
        
        system_prompt = self._system_prompt(**kwargs)
        context_prompt = self._context_prompt(**kwargs)
        # prepare full prompts:
        query = convert_to_list(query)
        full_prompts = [system_prompt + context_prompt + q for q in query]
        # tokenize the full prompts:
        input_ids = self.tokenizer.encode(full_prompts).to(self.config.device)
        # trunction:
        if self.config.truncate_length is not None:
            input_ids = self.__truncate_input_ids(input_ids, self.config.truncate_length, self.config.truncate_side)
        # padding:
        input_ids = self.__pad_input_ids(input_ids, self.config.pad_to_multiple_of, self.config.padding_side) 
        # generate the responses:
        responses: List[str] = self.generate(input_ids)
        
        return [
            {
                PromptType.SYSTEM: system_prompt,
                PromptType.CONTEXT: context_prompt,
                PromptType.QUERY: q,
                PromptType.RESPONSE: response,
                PromptType.PROMPT: full_prompt,
                PromptType.ALL: full_prompts + response,
            }
            for q, full_prompt, response in zip(query, full_prompts, responses)
        ]
        
    def generate(
        self,
        input_ids: List[torch.LongTensor],
    ) -> List[str]:
        """Generate the responses
        """
        batch_size = len(input_ids)
        generation_ids = [[] for _ in range(batch_size)]
        # reset kv cache
        self.model.reset_kv_cache()
        # generation loop:
        for _ in range(self.config.max_new_tokens):
            input_ids = torch.stack(
                [input_id for input_id in input_ids],
                dim=0,
            ).to(self.config.device)
            probs = self.model(input_ids=input_ids, temperature=self.config.temperature)
            # the shape of probs is (batch_size, vocab_size)
            sample_ids: List[torch.LongTensor] = []
            if self.config.decode_strategy == DecodeStrategy.GREEDY:
                sample_ids = self.__greedy_decode(probs)
            elif self.config.decode_strategy == DecodeStrategy.SAMPLING:
                sample_ids = self.__sampling_decode(probs)
            else:
                raise ValueError(f"Invalid decode strategy: {self.config.decode_strategy}, decode_strategy should be either GREEDY or SAMPLING")
            for idx, sample_id in enumerate(sample_ids):
                generation_ids[idx].append(sample_id)
            
            # catenate the sample ids to the input ids:
            for ibx in input_ids.shape[0]:
                input_ids[ibx] = torch.cat(
                    (
                        input_ids[ibx],
                        sample_ids[ibx]
                    ),
                    dim=0
                )
        
        # concatenate the generation_ids along the first dimension to convert
        # from List[List[torch.LongTensor]] to List[torch.LongTensor]
        concatenated_generation_ids = []
        for batch_generation_ids in generation_ids:
            concatenated_ids = torch.cat(batch_generation_ids, dim=0)
            concatenated_generation_ids.append(concatenated_ids)
        
        # decode the concatenated generation ids to strings
        return self.tokenizer.decode(concatenated_generation_ids)
    
    
    def __truncate_input_ids(
        self, 
        input_ids: List[torch.LongTensor],
        truncate_length: int,
        truncate_side: TruncateSide,
    ) -> List[torch.LongTensor]:
        """Truncate the input ids
        """
        for idx, input_id in enumerate(input_ids):
            if truncate_side == TruncateSide.LEFT:
                input_ids[idx] = input_id[-truncate_length:]
            elif truncate_side == TruncateSide.RIGHT:
                input_ids[idx] = input_id[:truncate_length]
            elif truncate_side == TruncateSide.MIDDLE:
                input_ids[idx] = torch.cat(
                    (
                        input_ids[idx][:truncate_length // 2],
                        input_ids[idx][-truncate_length // 2:]
                    ),
                    dim=0
                )
            else:
                raise ValueError(f"Invalid truncate side: {truncate_side}, truncate_side should be either LEFT, RIGHT, or MIDDLE")
        return input_ids
    
    
    def __pad_input_ids(
        self,
        input_ids: List[torch.LongTensor],
        pad_to_multiple_of: int,
        pad_side: PaddingSide,
    ) -> List[torch.LongTensor]:
        """Pad the input ids
        """
        max_input_length = max(input_id.shape[0] for input_id in input_ids)
        # ensure the padded length is a multiple of `pad_to_multiple_of`:
        padding_length = (max_input_length + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of
        for idx, input_id in enumerate(input_ids):
            to_be_padded = padding_length - input_id.shape[0]
            if pad_side == PaddingSide.LEFT:
                input_ids[idx] = F.pad(
                    input_id,
                    (to_be_padded, 0),
                    value=self.tokenizer.bos_id
                )
            elif pad_side == PaddingSide.RIGHT:
                input_ids[idx] = F.pad(
                    input_id,
                    (0, to_be_padded),
                    value=self.tokenizer.eos_id
                )
            else:
                raise ValueError(f"Invalid pad side: {pad_side}, pad_side should be either LEFT or RIGHT")
            
        return input_ids
        
    
    def __greedy_decode(
        self, 
        probs: torch.Tensor,
    ) -> List[torch.LongTensor]:
        """Greedy decode the input ids
        """
        selected_ids = []
        for prob in probs:
            # the shape of prob is (batch_size, vocab_size)
            select_id = torch.argmax(prob, dim=-1)
            selected_ids.append(select_id)
        return selected_ids
    
    def __sampling_decode(
        self,
        probs: torch.Tensor,
    ) -> torch.LongTensor:
        """Sampling decode the input ids
        """
        pass
     
    @staticmethod
    def load_generation_config(
        config_file: str, 
        **extra_configs
    ) -> InferenceConfig:
        """Load config from the original original Llama generation config
        
        Args:
            config_file(str): path to the config file of the original original Llama generation config in .json format
            extra_configs(dict, optional): extra (key, value) config pair(s), to overwrite `config.key = value`, \
                helpful to set some configurations that are neither fixed nor provided in the original config such as `device`, `seed`, etc.
                NOTE: if any required configuration is not found in the original config, you are supposed to pass it in `extra_configs`, \
                    otherwise, a `ValueError` will be raised.
        Returns:
            InferenceConfig: an InferenceConfig object initialized from the config file
        """
        raise NotImplementedError("TODO: Assignment5 - Task2")
    
    
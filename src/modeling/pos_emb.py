import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import apply_rotary_pos_emb


class NTKAwareRoPE(nn.Module):
    """NTK-aware RoPE module
    This is a series variants of the RoPE modules based on NTK theory to enhance its extrapolation ability.
    """
    
    def __init__(
        self, 
        dim: int, 
        max_seq_len: int,
        base: int = 10000,
        ratio: int = 1,
        dynamic: bool = False,
        dtype: torch.dtype = torch.float32,
        device: str = 'cpu',
    ) -> None:
        """Initialize NTK-aware RoPE Module
        
        Args:
            dim (int): The dimension of the RoPE
            max_seq_len (int): The maximum sequence length used in training
            base (int, optional): The base of the NTK. Defaults to 10000.
            ratio (int, optional): The ratio of the NTK. Defaults to 1.
            dynamic (bool, optional): Whether to use dynamic mode. Defaults to False.
            dtype (torch.dtype, optional): The dtype of the RoPE. Defaults to torch.float32.
            device (str, optional): The device of the RoPE. Defaults to 'cpu'.
        """
        super().__init__()
        
        self.dim = dim
        self.ms = max_seq_len
        self.ms_cached = max_seq_len * ratio
        self.base = base
        self.ratio = ratio
        self.is_dynamic = dynamic
        self.dtype = dtype
        self.device = device
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.set_cos_sin_cache(self.ms_cached, self.ratio)
        
    
    def set_cos_sin_cache(self, seq_len, ratio):
        """Set cos and sin compoent for RoPE

        Args:
            seq_len (int): The sequence length
        """
        t = torch.arange(seq_len, device = self.device, dtype = self.dtype)
        t = t / ratio
        
        freqs = torch.outer(t, self.inv_freq)
        # cos/sin vectors are with shape like :
        # [a_1, a_2, ..., a_{d/2}, a_1, a_2, ..., a_{d/2}] 
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(self.device), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(self.device), persistent=False)
    
    
    def forward(self, input: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """The forward pass of the NTK-aware RoPE module
        
        Args:
            input(torch.Tensor): input tensor, with shape: [batch_size, seq_len, num_heads, head_dim]
            offset(int, optional): The offset of the starting position index of the input tensor. Defaults to 0.
        
        Returns:
            output(torch.Tensor): embedded output tensor, with shape: [batch_size, seq_len, num_heads, head_dim]
         
        """
        input_dtype = input.dtype 
        seq_len = input.shape[1]
        cos = torch.rand((seq_len, input.shape[-1]))
        sin = torch.rand((seq_len, input.shape[-1]))
        if offset + seq_len > self.ms_cached:
            ratio_ = (offset + seq_len) // self.ms + 1
            if self.is_dynamic:
                # adjust self.ratio, self.ms_cached, invoke set_cos_sin_cache
                self.ratio = ratio_ if (ratio_ % 2 == 0) else (ratio_ + 1)
                self.ms_cached = self.ms * self.ratio
                self.set_cos_sin_cache(self.ms_cached, self.ratio)
            else :
                # give temporary result
                temp_ratio = ratio_ if (ratio_ % 2 == 0) else (ratio_ + 1)
                temp_seq_len = self.ms * temp_ratio
                t = torch.arange(temp_seq_len, device=self.device, dtype=self.dtype) / temp_ratio
                freqs = torch.outer(t, self.inv_freq)
                emb = torch.cat((freqs, freqs), dim = -1)
                cos, sin = emb.cos()[offset:offset+seq_len], emb.sin()[offset:offset+seq_len]
                return apply_rotary_pos_emb(input, cos, sin)
        
        cos, sin = self.cos_cached[offset:offset+seq_len], self.sin_cached[offset:offset+seq_len]
                    
        return apply_rotary_pos_emb(input, cos, sin).to(input_dtype)

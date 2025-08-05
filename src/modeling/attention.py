from typing import Optional, Tuple
from enum import Enum

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# from assignment1 implementations
from .norm import GroupRMSNorm


class AttnQKVPackFormat(Enum):
    QKV = "qkv_packed"
    Q_KV = "q_kv_packed"
    Q_K_V = "q_k_v_packed"


class AttnQKVLayout(Enum):
    BSHD = "bshd"
    SBHD = "sbhd"
    THD = "thd"

class AttnProjectionModule(nn.Module):
    """AttentProjectionModule
    This module is used to project the query, key, and value tensors according to layout and pack format.
    """
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        head_dim: int,
        num_q_head: int,
        num_kv_head: int,
        qkv_pack_format: AttnQKVPackFormat = AttnQKVPackFormat.Q_K_V,
        qkv_layout: AttnQKVLayout = AttnQKVLayout.BSHD,
        proj_init_seed: int = 42,
        proj_init_mean: float = 0.0,
        proj_init_std: float = 1.0,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        super().__init__()
        weights_kwargs = {'device': device, "dtype": dtype}
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_q_head = num_q_head
        self.num_kv_head = num_kv_head
        self.qkv_pack_format = qkv_pack_format
        self.qkv_layout = qkv_layout
        self.q_hidden_size = num_q_head * head_dim
        self.kv_hidden_size = num_kv_head * head_dim
        self.proj_init_seed = proj_init_seed
        self.proj_init_mean = proj_init_mean
        self.proj_init_std = proj_init_std
        self.proj_weights = nn.ParameterDict()
        # different weights for different pack format:
        if qkv_pack_format == AttnQKVPackFormat.Q_K_V:
            self.proj_weights['q_proj'] = nn.Parameter(torch.empty((self.q_hidden_size, hidden_size), **weights_kwargs))
            self.proj_weights['k_proj'] = nn.Parameter(torch.empty((self.kv_hidden_size, hidden_size), **weights_kwargs))
            self.proj_weights['v_proj'] = nn.Parameter(torch.empty((self.kv_hidden_size, hidden_size), **weights_kwargs))
        elif qkv_pack_format == AttnQKVPackFormat.Q_KV:
            self.proj_weights['q_proj'] = nn.Parameter(torch.empty((self.q_hidden_size, hidden_size), **weights_kwargs))
            self.proj_weights['kv_proj'] = nn.Parameter(torch.empty((2*self.kv_hidden_size, hidden_size), **weights_kwargs))
        elif qkv_pack_format == AttnQKVPackFormat.QKV:
            self.proj_weights['qkv_proj'] = nn.Parameter(torch.empty((self.q_hidden_size+2*self.kv_hidden_size, hidden_size), **weights_kwargs))
        else:
            raise ValueError(f"Invalid qkv_pack_format: {qkv_pack_format}")    
        self.reset_parameters()
     
    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat hidden_states n_rep times
        
        Args:
            hidden_states(torch.Tensor): hidden states tensor
            n_rep(int): number of repetitions
        Returns:
            torch.Tensor: repeated tensor
            change the shape of hidden_states from [b, s, hkv, d] to [b, s, hq, d]
            or [s, b, hkv, d] to [s, b, hq, d]
        """
        return torch.repeat_interleave(hidden_states, n_rep, dim=-2)   
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass
        
        Args:
            hidden_states(torch.Tensor): hidden states tensor
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: query, key, and value tensors
        """
        q, k, v = hidden_states, hidden_states, hidden_states
        if self.qkv_pack_format == AttnQKVPackFormat.Q_K_V:
            q = F.linear(q, self.proj_weights['q_proj'])
            k = F.linear(k, self.proj_weights['k_proj'])
            v = F.linear(v, self.proj_weights["v_proj"])
        elif self.qkv_pack_format == AttnQKVPackFormat.Q_KV:
            q = F.linear(q, self.proj_weights['q_proj'])
            kv = F.linear(k, self.proj_weights['kv_proj'])
            k, v = kv[:, :, :self.kv_hidden_size], kv[:, :, self.kv_hidden_size:]
        elif self.qkv_pack_format == AttnQKVPackFormat.QKV:
            qkv = F.linear(q, self.proj_weights['qkv_proj'])
            q, k, v = qkv[:, :, :self.q_hidden_size], qkv[:, :, self.q_hidden_size:self.q_hidden_size+self.kv_hidden_size], qkv[:, :, self.q_hidden_size+self.kv_hidden_size:]
        q = q.view(q.shape[0], q.shape[1], self.num_q_head, self.head_dim)
        k = k.view(k.shape[0], k.shape[1], self.num_kv_head, self.head_dim)
        v = v.view(v.shape[0], v.shape[1], self.num_kv_head, self.head_dim)
        return q, k, v 
   
    def reset_parameters(self):
        """Reset the parameters of the projection module"""
        torch.manual_seed(self.proj_init_seed + self.layer_idx + 1)
        for weight in self.proj_weights.values():
            nn.init.normal_(weight, mean=self.proj_init_mean, std=self.proj_init_std)
    
class OfflineSlidingWindowAttn(nn.Module):
    """Offline Sliding-Window Attention module
    This is a generalized variant of standard self-attention equipped with the sliding-window trick \
        to make use of spatial locality in language for computational efficiency, \
        with applying other methods to improve stability.
    """
    def __init__(
        self,
        head_dim: int,
        num_q_head: int,
        num_kv_head: int,
        qkv_pack_format: AttnQKVPackFormat = AttnQKVPackFormat.Q_K_V,
        qkv_layout: AttnQKVLayout = AttnQKVLayout.BSHD,
        window_size: Optional[int] = None,
        causal: bool = False,
        softmax_dropout_rate: float = 0.0,
        softmax_dropout_seed: int = 42,
        softmax_scale: Optional[float] = None,
        softmax_cap: Optional[float] = None,
        softmax_temp: float = 1.0,
        softmax_clip_range: Tuple[float, float] = (0., 1.),
        apply_qk_norm: bool = False,
        group_size: Optional[int] = None,
        eps: float = 1e-5,
        init_range: tuple = (-1.0, 1.0),
        init_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """Initialize Offline Sliding-Window Attention module
        
        Args:
            head_dim(int): head dimension size
            num_q_head(int): number of query heads
            num_kv_head(int): number of key/value heads
            qkv_pack_format(AttnQKVPackFormat, default = "q_k_v_packed"): qkv packed format
            qkv_layout(AttnQKVLayout, default = "bshd"): qkv shape layout
            window_size(int, default = None): window size
            causal(bool, default = False): if True, then apply causal masking as a prior to only allow unidirectional self-attention, otherwise bidirectional
            softmax_dropout_rate(float, default = 0.0): dropout probability for the softmax probs
            softmax_dropout_seed(int, default = 42): random seed for softmax drooput
            softmax_scale(float, default = None): softmax scale factor, if None, then applying the standard value: 1/√d
            softmax_cap(float, default = None): softmax capping to control the magnitude of the logits, if None, then NO capping is applied
            softmax_temp(float, default = 1.0): softmax temperature to control the sharpness of the distribution, only apply when softmax_cap is None
            softmax_clip_range(float, default = (0.0, 1.0): the range for softmax clipping to prevent the outliers from growing further
            apply_qk_norm(bool, default = False): if True, then apply qk norm
            group_size(int, optional, default = None): group size to split hidden size of query / key for GroupRMSNorm, if None, then set it to `head_dim`, if applying qk norm
            eps(float, default = 1e-5): epsilon for GroupRMSNorm, if applying qk norm
            init_range(tuple, default = (-1.0, 1.0)): the range of the initialization uniform distribution for GroupRMSNorm, if applying qk norm
            init_seed(int, default = 42): initialization seed for GroupRMSNorm, if applying qk norm
            dtype(torch.dtype, default = torch.float32): parameter dtype for GroupRMSNorm, if applying qk norm
            device(str, default = "cpu"): parameter device for GroupRMSNorm, if applying qk norm
        """
        super().__init__()
        # basic
        self.hd = head_dim
        self.hq = num_q_head
        self.hkv = num_kv_head
        self.qkv_pack_format = qkv_pack_format
        self.qkv_layout = qkv_layout
        self.window_size = window_size
        self.causal = causal
        # softmax clipping
        self.softmax_dropout = nn.Dropout(p=softmax_dropout_rate)
        self.softmax_dropout_seed = softmax_dropout_seed
        self.softmax_clip_range = softmax_clip_range
        # softmax stability
        self.softmax_scale = 1.0 / math.sqrt(self.hd) if softmax_scale is None else softmax_scale
        self.softmax_temp = softmax_temp
        self.softmax_cap = softmax_cap
        # RMSnorm for Q /KV
        self.apply_qk_norm = apply_qk_norm
        self.group_size = self.hd if group_size is None else group_size
        self.eps = eps
        self.norm_init_range = init_range
        self.norm_init_seed = init_seed
        self.dtype = dtype
        self.device = device
       
        if apply_qk_norm:
            self.queryRMSNormLayer = GroupRMSNorm(self.hd * self.hq, self.group_size, self.eps, 
                                         self.norm_init_range, self.norm_init_seed, self.dtype, self.device)
            self.keyRMSNormLayer = GroupRMSNorm(self.hd * self.hkv, self.group_size, self.eps, 
                                         self.norm_init_range, self.norm_init_seed, self.dtype, self.device)
        
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor = None,
        v: torch.Tensor = None,
        offset: int = 0,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The forward pass of Offline Sliding-Window Attention module
        
        Args:
            q(torch.Tensor): query tensor
            k(torch.Tensor): key tensor
            v(torch.Tensor): value tensor
            offset(int): offset for the query tensors (in complete implementation, kv_cache will give the offset) # TODO: abondon this argument
            cu_seqlens_q(Optional[torch.Tensor], default = None): cumulative sequence lengths for query tensor, with shape: [batch_size + 1, ]
            cu_seqlens_k(Optional[torch.Tensor], default = None): cumulative sequence lengths for key tensor, with shape: [batch_size + 1, ]
        Returns:
            torch.Tensor: output tensor o, with the same shape as q
        """
        q, k, v = q, self.repeat_kv(k, self.hq // self.hkv), self.repeat_kv(v, self.hq // self.hkv)
        # For each head, we apply the following steps:
        # 1. Q = GroupRMSNorm(Q), K = GroupRMSNorm(K)
        if self.apply_qk_norm:
            q = self.queryRMSNormLayer(q)
            k = self.keyRMSNormLayer(k)
        # After RMSNorm, the shape of q, k is [b ,s, hidden_size] or [s, b, hidden_size]
        # change the shape to [b, hq, s, hd]
        q = q.view(q.shape[0], q.shape[1], self.hq, self.hd)
        k = k.view(k.shape[0], k.shape[1], self.hq, self.hd)
        if self.qkv_layout == AttnQKVLayout.BSHD:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
        elif self.qkv_layout == AttnQKVLayout.SBHD:
            q = q.transpose(0, 2)
            k = k.transpose(0, 2)
            v = v.transpose(0, 2)
            
        # 2. calculate P = various_softmax(QK^T + mask)
        # if self.cap is None, use softmax temperature
        # P = scale * QK^T / temp + Mask
        P = q @ k.transpose(-2, -1)
        mask = torch.zeros_like(P)
        # if self.causal = False, apply normal sliding window mask:
        # q_i can only attend to k_j where j \in [i - window_size, i + window_size]
        # if self.causal = True, apply causal mask:
        # q_i can only attend to k_j where j \in [i - window_size, i]
        if self.window_size is None:
            self.window_size = float('inf')
        if not self.causal:
            for q_index in range(P.shape[-2]):
                for k_index in range(P.shape[-1]):
                    q_index += offset
                    if k_index > q_index + self.window_size or k_index < q_index - self.window_size:
                        mask[:, :, q_index, k_index] = -float('inf')
        else:
            for q_index in range(P.shape[-2]):
                for k_index in range(P.shape[-1]):
                    q_index += offset
                    if k_index > q_index or k_index < q_index - self.window_size:
                        mask[:, :, q_index, k_index] = -float('inf')
                        
        if self.softmax_cap is None:
            P = P * self.softmax_scale / self.softmax_temp + mask
        else:
            P = self.softmax_cap * torch.tanh(P * self.softmax_scale / self.softmax_temp) + mask
        
        # 3. A = safe_softmax_{row-wise}(P)
        # safe_softmax(x) = exp(x_i - max(x)) / sum(exp(x_j - max(x))) where i is the row index
        A = F.softmax(P - P.max(dim=-1, keepdim=True).values, dim=-1)
        
        # 4. A = dropout(clip((r-l)A +l), 0, 1)
        torch.manual_seed(self.softmax_dropout_seed)
        A = self.softmax_dropout(A)
        l,r = self.softmax_clip_range
        A = torch.clamp((r - l) * A + l, min=0, max=1)
        
        # 5. O = A @ V
        O = A @ v
        O = O.transpose(1, 2) if self.qkv_layout == AttnQKVLayout.BSHD else O.transpose(0, 2)
        return O
       
        
    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat hidden_states n_rep times
        
        Args:
            hidden_states(torch.Tensor): hidden states tensor
            n_rep(int): number of repetitions
        Returns:
            torch.Tensor: repeated tensor
            change the shape of hidden_states from [b, s, hkv, d] to [b, s, hq, d]
        """
        return torch.repeat_interleave(hidden_states, n_rep, dim=-2)
        
        
    
         
     
class OnlineSlidingWindowAttn(OfflineSlidingWindowAttn):
    """Online Sliding-Window Attention module
    This is a online version of Offline Sliding-Window Attention module \
        which only apply attention on a block of q, k, v in "bshd" layout and "q_k_v_packed" format \
            and update the global o with the local block of o using lse
    """
    def __init__(
        self,
        seqlen_q: int,
        seqlen_kv: int,
        block_size_q: int,
        block_size_kv: int,
        head_dim: int,
        num_q_head: int,
        num_kv_head: int,
        window_size: Optional[int] = None,
        causal: bool = False,
        softmax_scale: Optional[float] = None,
        softmax_cap: Optional[float] = None,
        softmax_temp: float = 1.0,
        apply_qk_norm: bool = False,
        group_size: Optional[int] = None,
        eps: float = 1e-5,
        init_range: tuple = (-1.0, 1.0),
        init_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """Initialize Online Sliding-Window Attention module
        
        Args:
            seqlen_q(int): the sequence length of q
            seqlen_kv(int): the sequence length of kv
            block_size_q(int): the block size of q
            block_size_kv(int): the block size of kv
            head_dim(int): head dimension size
            num_q_head(int): number of query heads
            num_kv_head(int): number of key/value heads
            window_size(int, default = None): window size
            causal(bool, default = False): if True, then apply causal masking as a prior to only allow unidirectional self-attention, otherwise bidirectional
            softmax_scale(float, default = None): softmax scale factor, if None, then applying the standard value: 1/√d
            softmax_cap(float, default = None): softmax capping to control the magnitude of the logits, if None, then NO capping is applied
            softmax_temp(float, default = 1.0): softmax temperature to control the sharpness of the distribution, only apply when softmax_cap is None
            apply_qk_norm(bool, default = False): if True, then apply qk norm
            group_size(int, optional, default = None): group size to split hidden size of query / key for GroupRMSNorm, if None, then set it to `head_dim`, if applying qk norm
            eps(float, default = 1e-5): epsilon for GroupRMSNorm, if applying qk norm
            init_range(tuple, default = (-1.0, 1.0)): the range of the initialization uniform distribution for GroupRMSNorm, if applying qk norm
            init_seed(int, default = 42): initialization seed for GroupRMSNorm, if applying qk norm
            dtype(torch.dtype, default = torch.float32): parameter dtype for GroupRMSNorm, if applying qk norm
            device(str, default = "cpu"): parameter device for GroupRMSNorm, if applying qk norm
        """
        super().__init__(
            head_dim=head_dim, # hd
            num_q_head=num_q_head, # hq
            num_kv_head=num_kv_head, # hkv
            window_size=window_size,
            causal=causal,
            softmax_scale=softmax_scale,
            softmax_cap=softmax_cap,
            softmax_temp=softmax_temp,
            apply_qk_norm=apply_qk_norm,
            group_size=group_size,
            eps=eps,
            init_range=init_range,
            init_seed=init_seed,
            dtype=dtype,
            device=device,
        )
        # OSWA attribute:
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.block_size_q = block_size_q
        self.block_size_kv = block_size_kv
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        global_o: torch.Tensor,
        global_lse: torch.Tensor,
        block_idx_q: int,
        block_idx_kv: int,
    ) -> None:
        """The forward pass of Offline Sliding-Window Attention module
        
        Args:
            q(torch.Tensor): query tensor, with shape: [batch_size, block_size_q, num_q_head, head_dim]
            k(torch.Tensor): key tensor, with shape: [batch_size, block_size_kv, num_kv_head, head_dim]
            v(torch.Tensor): value tensor, with shape: [batch_size, block_size_kv, num_kv_head, head_dim]
            global_o(torch.Tensor): global output tensor to be updated inplace, with shape: [batch_size, seqlen_q, num_q_head, head_dim]
            global_lse(torch.Tensor): global lse tensor to be updated inplace, with shape: [batch_size, num_q_head, seqlen_q]
            block_idx_q(int): the block index of q
            block_idx_kv(int): the block index of kv
        """
        q, k, v = self.get_qkv(q, k, v)
        # For each head, we apply the following steps:
        # 1. Q = GroupRMSNorm(Q), K = GroupRMSNorm(K)
        if self.apply_qk_norm:
            queryRMSNormLayer = GroupRMSNorm(self.hd * self.hq, self.group_size, self.eps, 
                                         self.norm_init_range, self.norm_init_seed, self.dtype, self.device)
            keyRMSNormLayer = GroupRMSNorm(self.hd * self.hkv, self.group_size, self.eps, 
                                         self.norm_init_range, self.norm_init_seed, self.dtype, self.device)
            query = queryRMSNormLayer(q)
            key = keyRMSNormLayer(k)
        else:
            query = q
            key = k
        
        # 2. get attention scores P = scale * QK^T / temp + mask
        P = query @ key.transpose(-2, -1)
        mask = torch.zeros_like(P)
        for local_q_index in range(self.block_size_q):
            for local_k_index in range(self.block_size_kv):
                global_q_index = block_idx_q * self.block_size_q + local_q_index
                global_k_index = block_idx_kv * self.block_size_kv + local_k_index
                if global_k_index > global_q_index + self.window_size or global_k_index < global_q_index - self.window_size:
                    mask[:, :,local_q_index, local_k_index] = -float('inf')
                if self.causal:
                    if global_k_index > global_q_index:
                        mask[:, :,local_q_index, local_k_index] = -float('inf')
                    
        if self.softmax_cap is None:
            P = P * self.softmax_scale / self.softmax_temp + mask
        else:
            P = self.softmax_cap * torch.tanh(P * self.softmax_scale / self.softmax_temp) + mask
        
        # 3. calculate p_local and o_local
        # the shape of P is [b, hq, bq, bkv]
        # the shape of global_lse is [b, hq, sq]
        q_start = block_idx_q * self.block_size_q
        q_end = q_start + self.block_size_q
        local_lse = torch.logsumexp(P, dim=-1)
        # the shape of local_lse is [b, hq, bq]
        
        p_local = torch.exp(P - local_lse.unsqueeze(-1))
        o_local = p_local @ v
        # the shape of o_local is [b, hq, bq, d]
        
        # 4. update global_o and global_lse
        # the shape of global_o is [b, s, h, d]
        old_lse = global_lse[:, :, q_start:q_end]
        lse_max = torch.max(old_lse, local_lse)
        lse_min = torch.min(old_lse, local_lse)
        new_lse = lse_max + torch.log1p(torch.exp(lse_min - lse_max))
        
        old_c = torch.exp(old_lse - new_lse).unsqueeze(-1)
        new_c = torch.exp(local_lse - new_lse).unsqueeze(-1)
        old_o = global_o.transpose(1, 2)[:, :, q_start:q_end, :] # [b, h, bq, d]
        # update
        o_new_block = old_o * old_c + o_local * new_c # [b, h, bq, d]
        global_o[:, q_start:q_end,:, :] = o_new_block.transpose(1, 2)
        global_lse[:, :, q_start:q_end] = new_lse
        
        

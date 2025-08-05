from typing import Optional, Tuple, Sequence, Literal, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup

# from assignment1 implementations
from .vocab_emb import ParallelVocabEmbedding
from .pos_emb import NTKAwareRoPE
from .norm import GroupRMSNorm

# from assignment2 implementations
from .mlp import (
    MLPActivationType,
    DenseMLPWithLoRA,
    SparseMLPWithLoRA,
)

# from assignment3 implementations
from .attention import (
    AttnQKVPackFormat,
    AttnQKVLayout,
    AttnProjectionModule,
    OfflineSlidingWindowAttn,
    OnlineSlidingWindowAttn,
)

from .config import (
    BaseConfig,
    config_dataclass,
    make_required_field,
    make_fixed_field,
)


@config_dataclass
class TransformerConfig(BaseConfig):
    """Transformer Configurations Dataclass"""
    
    # common transformer configurations
    num_layers: int = make_required_field()
    hidden_size: int = make_required_field()
    ffh_size: int = make_required_field()
    max_seq_len: int = make_required_field()
    param_dtype: torch.dtype = torch.float32
    param_device: str = "cpu"
    init_base_seed: int = 42
    
    # fixed distributed configurations
    rank: int = make_fixed_field(0)
    world_size: int = make_fixed_field(1)
    process_group: Optional[ProcessGroup] = make_fixed_field(None)
    
    # vocab embedding configurations
    vocab_size: int = make_required_field()
    vocab_init_mean: float = 0.0
    vocab_init_std: float = 1.0
    
    # positional embedding configurations
    rope_base: int = 10000
    rope_ratio: int = 1
    rope_dynamic: bool = False
    
    # normalization configurations
    group_size: Optional[int] = None
    eps: float = 1e-5
    norm_init_range: tuple = (-1.0, 1.0)
    
    # projection configurations
    proj_init_seed: int = 42
    proj_init_mean: float = 0.0
    proj_init_std: float = 1.0
    lm_head_tied: bool = False
    
    # attention configurations
    online_attn_block_size: Optional[int] = None # NOTE: if None, then use offline mode, otherwise use online mode
    head_dim: int = make_required_field()
    num_q_head: int = make_required_field()
    num_kv_head: int = make_required_field()
    qkv_pack_format: AttnQKVPackFormat = AttnQKVPackFormat.Q_K_V
    qkv_layout: AttnQKVLayout = AttnQKVLayout.BSHD
    window_size: Optional[int] = None
    causal: bool = False
    softmax_dropout_rate: float = 0.0
    softmax_dropout_seed: int = 42
    softmax_scale: Optional[float] = None
    softmax_cap: Optional[float] = None
    softmax_temp: float = 1.0
    softmax_clip_range: Tuple[float, float] = (0., 1.)
    apply_qk_norm: bool = False
    qk_norm_group_size: Optional[int] = None # NOTE: the other configurations of qk norm are the same as the ones of normalization above
    
    # dense mlp configurations
    activation_type: MLPActivationType = MLPActivationType.SILU
    lora_rank: int = 0
    lora_alpha: Optional[float] = None
    lora_dropout_rate: float = 0.0
    lora_dropout_seed: int = 42
    lora_init_base_seed: int = 42
    
    # sparse mlp configurations (optional)
    num_experts: Optional[int] = None # NOTE: if None, then use dense mlp, otherwise use sparse mlp
    moe_topk: int = 1
    gate_init_mean: float = 0.0
    gate_init_std: float = 1.0


class TransformerDecoderKVCache(nn.Module):
    """Transformer KV cache module
    This is a simple module to manage cached past key-value pairs for each transformer decoder layer \
        tradeoff memory footprint for avoiding redundant computation during inference.
    """
    def __init__(
        self,
        qkv_layout: AttnQKVLayout = AttnQKVLayout.BSHD,
        num_layers: int = 1,
    ):
        """Initialize Transformer KV cache module
        
        Args:
            qkv_layout (AttnQKVLayout, optional): Layout of the q, k, v tensors. Defaults to AttnQKVLayout.BSHD.
            num_layers (int, optional): Number of transformer layers. Defaults to 1.
        """
        super().__init__()
        # just implement BSHD layout
        self.num_layers = num_layers
        self.key_cache: Dict[int, torch.Tensor] = {}
        self.value_cache: Dict[int, torch.Tensor] = {}
        self.qkv_layout = qkv_layout

    def has(self, layer_idx: int) -> bool:
        """Check if cached past key-value pairs exist for a specific layer
        
        Args:
            layer_idx (int): Layer index

        Returns:
            bool: True if cached past key-value pairs exist for the layer, False otherwise
        """

        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"Invalid layer index: {layer_idx}, should be in the range of [0, {self.num_layers})")
        return layer_idx in self.key_cache and layer_idx in self.value_cache
        
        
    def get(
        self, 
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Get cached past key-value pairs with their optional cumulative sequence lengths for a specific layer
        
        Args:
            layer_idx (int): Layer index

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: (k, v, optional cu_seqlens)
            
        Raises:
            KeyError: If cached past key-value pairs do not exist for the layer
        """
        if not self.has(layer_idx):
            return None, None, None
        return (self.key_cache[layer_idx], self.value_cache[layer_idx], None)
   
    
    def set(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> None:
        """Set cached past key-value pairs with their optional cumulative sequence lengths for a specific layer
        
        Args:
            layer_idx (int): Layer index
            k (torch.Tensor): Key tensor to set
            v (torch.Tensor): Value tensor to set
            cu_seqlens (Optional[torch.Tensor], optional): Cumulative sequence lengths for the key-value pairs to set. Defaults to None.
            NOTE: The `cu_seqlens` must be provided if the `qkv_layout` is AttnQKVLayout.THD
        """
        # just implement BSHD layout
        # ignore cu_seqlens
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"Invalid layer index: {layer_idx}, should be in the range of [0, {self.num_layers})")
        self.key_cache[layer_idx] = k
        self.value_cache[layer_idx] = v
        

    def append(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> None:
        """Dynamically append current cached past key-value pairs with their optional cumulative sequence lengths to the existing ones for a specific layer
        
        Args:
            layer_idx (int): Layer index
            k (torch.Tensor): Key tensor to append
            v (torch.Tensor): Value tensor to append
            cu_seqlens (Optional[torch.Tensor], optional): Cumulative sequence lengths for the key-value pairs to append. Defaults to None.
            NOTE: The `cu_seqlens` must be provided if the `qkv_layout` is AttnQKVLayout.THD, \
                and all of the pass-in arguments should be consistent with the existing ones.
        """
        # ignore cu_seqlens
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"Invalid layer index: {layer_idx}, should be in the range of [0, {self.num_layers})")
        # Be sure that the shape of k and v are [b, 1, h, d]
        if not self.has(layer_idx):
            self.set(layer_idx, k, v)
        else:
            if self.qkv_layout == AttnQKVLayout.BSHD:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], k], dim=1)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], v], dim=1)
            elif self.qkv_layout == AttnQKVLayout.SBHD:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], k], dim=0)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], v], dim=0)
            elif self.qkv_layout == AttnQKVLayout.THD:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], k.squeeze(0)], dim=0)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], v.squeeze(0)], dim=0)
    
    def reset(self):
        """Clear the cache memory and reset to the initial state
        """
        self.key_cache.clear()
        self.value_cache.clear()


class TransformerDecoderAttnLayer(nn.Module):
    """Transformer Decoder Attention Layer module
    This is a variant of transformer decoder attention layer, consisting of a RMSnorm layer, \
        a qkv projection layer, a ntk-aware rope layer, an attention layer and a linear projection for O.
    """
    def __init__(
        self,
        config: TransformerConfig,
        layer_idx: int = 0,
    ):
        super().__init__()
        weights_kwargs = {'device': config.param_device, "dtype": config.param_dtype}
        self.config = config
        self.layer_idx = layer_idx    
        # initialize the sub-layers
        ### self-attention layer :
        # RMSnorm layer for hidden states:
        self.attn_RMSnorm = GroupRMSNorm(
            hidden_size=config.hidden_size,
            group_size=config.group_size,
            eps=config.eps,
            init_range=config.norm_init_range,
            init_seed=config.init_base_seed + layer_idx + 1,
            dtype=config.param_dtype,
            device=config.param_device
        )
        # qkv projection layer:
        # for attention like MQA or GQA, the hidden_size of K, V is different from the hidden_size of Q
        self.attn_proj_layer = AttnProjectionModule(
            layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_q_head=config.num_q_head,
            num_kv_head=config.num_kv_head,
            qkv_pack_format=config.qkv_pack_format,
            qkv_layout=config.qkv_layout,
            proj_init_seed=config.proj_init_seed,
            proj_init_mean=config.proj_init_mean,
            proj_init_std=config.proj_init_std,
            dtype=config.param_dtype,
            device=config.param_device
        )
        # ntk-aware rope layer:
        self.rope_layer = NTKAwareRoPE(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
            ratio=config.rope_ratio,
            dynamic=config.rope_dynamic,
            dtype=config.param_dtype,
            device=config.param_device
        )
        # attention layer:
        self.attn_module = OfflineSlidingWindowAttn(
            head_dim=config.head_dim,
            num_q_head=config.num_q_head,
            num_kv_head=config.num_kv_head,
            qkv_pack_format=config.qkv_pack_format,
            qkv_layout=config.qkv_layout,
            window_size=config.window_size,
            causal=config.causal,
            softmax_dropout_rate=config.softmax_dropout_rate,
            softmax_dropout_seed=config.softmax_dropout_seed,
            softmax_scale=config.softmax_scale,
            softmax_cap=config.softmax_cap,
            softmax_temp=config.softmax_temp,
            softmax_clip_range=config.softmax_clip_range,
            apply_qk_norm=config.apply_qk_norm,
            group_size=config.qk_norm_group_size,
            eps=config.eps,
            init_range=config.norm_init_range,
            init_seed=config.init_base_seed + layer_idx + 2,
            dtype=config.param_dtype,
            device=config.param_device
        )
        if config.online_attn_block_size is not None:
            self.attn_module = OnlineSlidingWindowAttn(
                seqlen_q=config.max_seq_len,
                seqlen_kv=config.max_seq_len,
                block_size_q=config.online_attn_block_size,
                block_size_kv=config.online_attn_block_size,
                head_dim=config.head_dim,
                num_q_head=config.num_q_head,
                num_kv_head=config.num_kv_head,
                window_size=config.window_size,
                causal=config.causal,
                softmax_scale=config.softmax_scale,
                softmax_cap=config.softmax_cap,
                softmax_temp=config.softmax_temp,
                apply_qk_norm=config.apply_qk_norm,
                group_size=config.qk_norm_group_size,
                eps=config.eps,
                init_range=config.norm_init_range,
                init_seed=config.init_base_seed + layer_idx + 2,
                dtype=config.param_dtype,
                device=config.param_device
            )
        # linear projection for O:
        self.o_proj = nn.Parameter(torch.empty((config.hidden_size, config.hidden_size), **weights_kwargs))
        self.reset_parameters()
         
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        kv_cache: Optional[TransformerDecoderKVCache] = None,
    ) -> torch.Tensor:
        """The forward pass of Transformer Decoder Attention Layer module
        """
        residual = hidden_states
        # Group RMS Normalization
        hidden_states = self.attn_RMSnorm(hidden_states)
        # QKV Projection
        q, k, v = self.attn_proj_layer(hidden_states) #[b, s, h, hd]
        # Apply RoPE
        # NOTE: RoPE can only process BSHD layout
        offset = 0
        query, key, value = q, k, v
        if kv_cache is not None:
            if kv_cache.has(self.layer_idx):
                # has kv cache, we do the following:
                # 1. get past_key_value
                k_cache, v_cache, cu_seqlens = kv_cache.get(self.layer_idx)
                # 2. get offset, and do RoPE on query and key
                if self.config.qkv_layout == AttnQKVLayout.SBHD:
                    offset = k_cache.shape[0]
                    q, k, v = self.rope_layer(q, offset).transpose(0, 1), self.rope_layer(k, offset).transpose(0, 1), v.transpose(0, 1)
                elif self.config.qkv_layout == AttnQKVLayout.BSHD:
                    offset = k_cache.shape[1]
                    q, k = self.rope_layer(q, offset), self.rope_layer(k, offset)
                # 3. append to kv_cache
                kv_cache.append(self.layer_idx, k, v, None)
            else:
                # no kv cache in this layer, we do the following:
                # 1. do RoPE on query and key
                if self.config.qkv_layout == AttnQKVLayout.SBHD:
                    q, k = self.rope_layer(q, 0).transpose(0, 1), self.rope_layer(k, 0).transpose(0, 1)
                elif self.config.qkv_layout == AttnQKVLayout.BSHD:
                    q, k = self.rope_layer(q, 0), self.rope_layer(k, 0)
                # 2. set to kv_cache
                kv_cache.set(self.layer_idx, k, v, None)
            
            query, key, value = q, kv_cache.get(self.layer_idx)[0], kv_cache.get(self.layer_idx)[1] # consistent with qkv_layout
        else:
            # kv_cache unavailable
            if self.config.qkv_layout == AttnQKVLayout.SBHD:
                q, k = self.rope_layer(q, 0).transpose(0, 1), self.rope_layer(k, 0).transpose(0, 1)
            elif self.config.qkv_layout == AttnQKVLayout.BSHD:
                q, k = self.rope_layer(q, 0), self.rope_layer(k, 0)
            query, key, value = q, k, v
        # Apply attention
        # TODO: add OnlineSlidingWindowAttn
        o = self.attn_module(query, key, value, offset)
        # Reshape to [b, s, h]
        if self.config.qkv_layout == AttnQKVLayout.SBHD:
            o = o.transpose(0, 1)
        o = o.reshape(o.shape[0], o.shape[1], -1)
        # Apply linear projection
        o = F.linear(o, self.o_proj)
        # Add residual connection
        output = o + residual
        return output
        
    def reset_parameters(self):
        """Initialize learnable parameters for Transformer Decoder Attention Layer module"""
        torch.manual_seed(self.config.proj_init_seed + self.layer_idx + 2)
        nn.init.normal_(self.o_proj, self.config.proj_init_mean, self.config.proj_init_std)
    

class TransformerDecoderMLPLayer(nn.Module):
    """Transformer Decoder MLP Layer module
    This is a variant of transformer decoder mlp layer, consisting of a RMSnorm layer, \
        a dense / sparse feed-forward mlp layer, supporting LoRA adaption intrinsically.
    """
    def __init__(
        self,
        config: TransformerConfig,
        layer_idx: int = 0,
    ):
        super().__init__()
        weights_kwargs = {'device': config.param_device, "dtype": config.param_dtype}
        self.config = config
        self.layer_idx = layer_idx
        # initialize the sub-layers
        # RMSnorm layer:
        self.mlp_RMSnorm = GroupRMSNorm(
            hidden_size=config.hidden_size,
            group_size=config.group_size,
            eps=config.eps,
            init_range=config.norm_init_range,
            init_seed=config.init_base_seed + layer_idx + 3,
            dtype=config.param_dtype,
            device=config.param_device
        )
        # dense / sparse feed-forward mlp layer:
        if config.num_experts is None:
            self.mlp = DenseMLPWithLoRA(
                hidden_size=config.hidden_size,
                ffh_size=config.ffh_size,
                activation_type=config.activation_type,
                init_base_seed=config.init_base_seed + layer_idx + 4,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout_rate=config.lora_dropout_rate,
                lora_dropout_seed=config.lora_dropout_seed + layer_idx,
                lora_init_base_seed=config.lora_init_base_seed + layer_idx,
                dtype=config.param_dtype,
                device=config.param_device
            )
        else:
            self.mlp = SparseMLPWithLoRA(
                hidden_size=config.hidden_size,
                ffh_size=config.ffh_size,
                activation_type=config.activation_type,
                num_experts=config.num_experts,
                moe_topk=config.moe_topk,
                rank=config.rank,
                world_size=config.world_size,
                process_group=config.process_group,
                init_mean=config.gate_init_mean,
                init_std=config.gate_init_std,
                init_base_seed=config.init_base_seed + layer_idx + 4,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout_rate=config.lora_dropout_rate,
                lora_dropout_seed=config.lora_dropout_seed + layer_idx,
                lora_init_base_seed=config.lora_init_base_seed + layer_idx,
                dtype=config.param_dtype,
                device=config.param_device
            )
            
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        # Group RMS Normalization
        hidden_states = self.mlp_RMSnorm(hidden_states)
        # Apply dense / sparse feed-forward mlp
        hidden_states = self.mlp(hidden_states)
        # Add residual connection
        output = hidden_states + residual
        return output
        
class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer module
    This is a variant of transformer decoder layer, consisting of two sub-layers: \
            one offline / online self-attention layer, along with qkv projection, ntk-aware rope and out projection, \
            and one dense / sparse feed-forward mlp layer, supporting LoRA adaption intrinsically, \
        which are concatenated sequentially with residual connections and group rms normalization.
    """
    # NOTE: only implement BSHD layout
    def __init__(
        self,
        config: TransformerConfig,
        layer_idx: int = 0,
    ):
        """Initialize Transformer Decoder Layer module
        
        Args:
            config (TransformerConfig): transformer configuration
            layer_idx (int): layer index, in the range of [0, num_layers). Defaults to 0.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.attention_layer = TransformerDecoderAttnLayer(config, layer_idx)
        self.mlp_layer = TransformerDecoderMLPLayer(config, layer_idx)
             
    def forward(
        self,
        input: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        kv_cache: Optional[TransformerDecoderKVCache] = None,
    ) -> torch.Tensor:
        """The forward pass of Transformer Decoder Layer module
        
        Args:
            input(torch.Tensor): input hidden states tensor, with shape: [batch_size, seq_len, hidden_size]
            cu_seqlens(torch.Tensor, optional): cumulative sequence lengths for input tensor, with shape: [inner_batch_size + 1, ]
            kv_cache(Optional[TransformerDecoderKVCache], default = None): transformer kv cache, to retrieve / update past key and value during inference, \
                if None, then no kv cache (i.e. during training)
            NOTE: if `cu_seqlens` is not None, then the `batch_size` in the shape of `input` is ensured to be `1` to remain the 3-dim shape, \
                while the real `batch_size` is inferred from `cu_seqlens` (i.e. `inner_batch_size`) since the inner sequences are concatenated along the `seqlen` dim.
        Returns:
            torch.Tensor: output hidden states tensor, with the same shape as input
        """
        o = self.attention_layer(input, cu_seqlens, kv_cache)
        output = self.mlp_layer(o)
        return output
    
    
    # def reset_parameters(self):
    #     """Initialize learnable parameters for Transformer Decoder Layer module"""
    #     raise NotImplementedError("TODO: Assignment4 - Task2")


class TransformerDecoderBlock(nn.Module):
    """Transformer Decoder Block module
    
    This is a standard decoder-only transformer block for language modeling, \
        which mainly consists of a sequence of transformer decoder layers, \
        transforming the hidden states of input token ids initialized from vocab embedding, \
        and finally returning the vocab logits with a lm head projection.
    """
    
    def __init__(
        self,
        config: TransformerConfig,
    ):
        """Initialize Transformer Decoder Block module
        
        Args:
            config (TransformerConfig): transformer configuration
        """
        super().__init__()
        self.config = config
        # weights_kwargs = {'device': config.param_device, "dtype": config.param_dtype}
        # kv_cahce
        self.kv_cache = TransformerDecoderKVCache(
            qkv_layout=config.qkv_layout,
            num_layers=config.num_layers
        )
        # ParallelVocabEmbedding
        self.vocab_embedding = ParallelVocabEmbedding(
            vocab_size=config.vocab_size,
            emb_size=config.hidden_size,
            rank=config.rank,
            world_size=config.world_size,
            process_group=None,
            init_mean=config.vocab_init_mean,
            init_std=config.vocab_init_std,
            init_base_seed=config.init_base_seed,
            dtype=config.param_dtype,
            device=config.param_device
        )
        # DecoderLayers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_layers)
        ])
        # FinalNorm
        self.final_norm = GroupRMSNorm(
            hidden_size=config.hidden_size,
            group_size=config.group_size,
            eps=config.eps,
            init_range=config.norm_init_range,
            init_seed=config.init_base_seed, 
            dtype=config.param_dtype,
            device=config.param_device
        )
        # LmHead
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, device=config.param_device, dtype=config.param_dtype)
        if self.config.lm_head_tied:
            self.lm_head.weight = self.vocab_embedding.weight
        self.reset_parameters()
        
        print(f"TransformerDecoderBlock initialized with {self.num_parameters()} parameters, with {self.num_parameters(True)} learnable parameters")
        print(f"TransformerDecoderBlock memory footprint: {self.num_memory_footprint()} B")
         
    def forward(
        self,
        input_ids: torch.LongTensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The forward pass of Transformer Decoder Block module
        
        Args:
            input_ids(torch.LongTensor): the vocab ids for the input, with shape: [batch_size, seq_len]
            cu_seqlens(torch.Tensor, optional): cumulative sequence lengths, with shape: [inner_batch_size + 1, ]
            NOTE: if `cu_seqlens` is not None, then the `batch_size` in the shape of `input_ids` is ensured to be `1` to remain the 2-dim shape, \
                while the real `batch_size` is inferred from `cu_seqlens` (i.e. `inner_batch_size`) since the inner sequences are concatenated along the `seqlen` dim.
        Returns:
            torch.Tensor: output tensor as vocab logits, with shape: [batch_size, seq_len, vocab_size]
        """
        # Embedding
        # NOTE: world_size is fixed to 1. Only one ParallelVocabEmbedding block is used.
        hidden_states = self.vocab_embedding(input_ids)
        # DecoderLayers
        # in training mode, kv_cache is forbidden
        kv_cache = None if self.training else self.kv_cache
        for layer in self.decoder_layers:
            hidden_states = layer(hidden_states, cu_seqlens, kv_cache)
        # FinalNorm
        hidden_states = self.final_norm(hidden_states)
        # LmHead
        logits = self.lm_head(hidden_states)
        return logits
    
    def get_kv_cache(self) -> TransformerDecoderKVCache:
        """Get the TransformerDecoderKVCache object managing the kv cache memory"""
        return self.kv_cache
    
    def set_kv_cache(self, kv_cache: TransformerDecoderKVCache):
        """Set the TransformerDecoderKVCache object managing the kv cache memory"""
        self.kv_cache = kv_cache
    
    def reset_kv_cache(self):
        """Clear the cache memory and reset to the initial state"""
        self.kv_cache.reset()
       
    def reset_parameters(self):
        """Initialize learnable parameters for Transformer Decoder Block module"""
        if not self.config.lm_head_tied:
            torch.manual_seed(self.config.proj_init_seed)
            nn.init.normal_(self.lm_head.weight.data, self.config.proj_init_mean, self.config.proj_init_std)
        
     
    def num_parameters(
        self,
        learnable_only: bool = False, 
        unit: Literal["1", "K", "M", "B"] = "1"
    ) -> float:
        """Compute the number of (learnable) parameters in the Llama Model module
        
        Args:
            learnable_only(bool, optional): whether to count only learnable parameters or not, default to False
            unit(str, optional): unit of the number of parameters, default to '1' for "1", \
                other options include 'K' for "1 thousand", 'M' for "1 million", 'B' for "1 billion"
        Returns:
            float: the number of (learnable) parameters in the Llama Model module in the specified unit
        """
        total_params = sum(p.numel() for p in self.parameters())
        if learnable_only:
            total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if unit == "1":
            return total_params
        elif unit == "K":
            return total_params / 1000
        elif unit == "M":
            return total_params / 1000000
        elif unit == "B":
            return total_params / 1000000000
        else:
            raise ValueError(f"Invalid unit: {unit}")
    
    def num_memory_footprint(
        self,
        unit: Literal["B", "KB", "MB", "GB"] = "B"
    ) -> float:
        """Compute the theoretical memory footprint of the Llama Model module's parameters
        
        Args:
            unit(str, optional): unit of the memory footprint, default to 'B' for "1 byte", \
                other options include 'KB' for "1 kilobyte", 'MB' for "1 megabyte", 'GB' for "1 gigabyte"
                
        Returns:
            float: the theoretical memory footprint of the Llama Model module's parameters in the specified unit
        """
        total_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        divisior = {
            "B": 1,
            "KB": 1024,
            "MB": 1024 * 1024,
            "GB": 1024 * 1024 * 1024
        }
        return total_bytes / divisior[unit]

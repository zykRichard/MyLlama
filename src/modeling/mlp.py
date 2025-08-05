from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup

ACT2FN = {
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
    "sigmoid": F.sigmoid,
    "bilinear": F.bilinear
}

class MLPActivationType(Enum):
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    SIGMOID = "sigmoid"
    BILINEAR = "bilinear"


class DenseMLPWithLoRA(nn.Module):
    """Dense MLP module with LoRA adapters
    This is a GLU-style dense MLP layer with LoRA adapters.
    """
    
    def __init__(self,
        hidden_size: int,
        ffh_size: int,
        activation_type: MLPActivationType = MLPActivationType.SILU,
        init_base_seed: int = 42,
        lora_rank: int = 0,
        lora_alpha: Optional[float] = None,
        lora_dropout_rate: float = 0.0,
        lora_dropout_seed: int = 42,
        lora_init_base_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """Initialize Dense MLP module with LoRA adapters
        Args:
            hidden_size(int): hidden dimension size
            ffh_size(int): hidden dimension size
            activation_type(MLPActivationType, default = "silu"): activation type
            init_base_seed(int, default = 42): seed for base weight initialization
            lora_rank(int, default = 0): lora rank, if 0, then no lora to apply
            lora_alpha(Optional[float], default = None): lora alpha, if None, then set to lora_rank
            lora_dropout_rate(float, default = 0.0): lora dropout rate
            lora_dropout_seed(int, default = 42): lora dropout seed
            lora_init_base_seed(int, default = 42): seed for lora weight initialization
            dtype(torch.dtype, default = torch.float32): parameter dtype
            device(str, default = "cpu"): parameter device
        """
        super().__init__()
        weights_kwargs = {'device': device, "dtype": dtype}
        self.hidden_size = hidden_size
        self.ffh_size = ffh_size
        # MLP linear projection weight
        # Attention that F.linear calculate the transpose of the weight
        # Record the linear projection weight by transpose format
        self.gate_proj = nn.Parameter(torch.empty((self.ffh_size, self.hidden_size), **weights_kwargs))
        self.up_proj = nn.Parameter(torch.empty((self.ffh_size, self.hidden_size), **weights_kwargs))
        self.down_proj = nn.Parameter(torch.empty((self.hidden_size, self.ffh_size), **weights_kwargs))
        # MLP activation function
        self.activation_type = activation_type
        self.actfn = ACT2FN[activation_type.value]
        
        self.init_base_seed = init_base_seed
        self.lora_rank = lora_rank
        self.dtype = dtype
        self.device = device
        # loRA Related:
        if self.lora_rank > 0:
            self.lora_init_base_seed = lora_init_base_seed
            self.lora_dropout_seed = lora_dropout_seed
            self.lora_alpha = lora_rank if lora_alpha is None else lora_alpha
            self.lora_dropout_rate = lora_dropout_rate
            # LoRA weight
            self.lora_A = nn.Parameter(torch.empty((self.lora_rank, self.hidden_size), **weights_kwargs))
            self.lora_B = nn.Parameter(torch.empty((self.hidden_size, self.lora_rank), **weights_kwargs))
            self.dropout = nn.Dropout(p=self.lora_dropout_rate)
        
        self.reset_parameters() 
        
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward pass of the Dense MLP module with LoRA adapters
        
        Args:
            input(torch.Tensor): input tensor, with shape: [batch_size, seq_len, hidden_size]
            
        Returns:
            output(torch.Tensor): output tensor, with shape: [batch_size, seq_len, hidden_size]
        """
        kwargs = {"dtype": input.dtype, "device": input.device}
        input = input.float().to(self.device) 
        # MLP process :
        # MLP(x) = (Fn(xW_gate) * xW_up)W_down
        action_state = self.actfn(F.linear(input, self.gate_proj, bias=None))
        up_state = F.linear(input, self.up_proj, bias=None)
        gated_state = action_state * up_state
        MLPstate = F.linear(gated_state, self.down_proj, bias=None)
        
        # LoRA process :
        # LoRA(x) = Dropout(scale * xAB)
        if self.lora_rank > 0:
            scale = self.lora_alpha / self.lora_rank
            torch.manual_seed(self.lora_dropout_seed)
            LoRAstate = scale * F.linear(F.linear(self.dropout(input), self.lora_A), self.lora_B)
            MLPstate = MLPstate + LoRAstate
        
        return MLPstate.to(**kwargs)
    
    def reset_parameters(self):
        """Initialize the weights of the Dense MLP module with LoRA adapters
        from a normal distribution (or a uniform distribution for lora weights)
        """
        # MLP weight initialize:
        init_func = nn.init.kaiming_normal_
        init_kwargs = {"mode": "fan_in", "nonlinearity": "relu"}
        if self.activation_type == MLPActivationType.SIGMOID or \
            self.activation_type == MLPActivationType.BILINEAR:
            init_func = nn.init.xavier_uniform_
            init_kwargs = {"gain": 1.0}
            
        MLPweights = [self.up_proj, self.gate_proj, self.down_proj]
        for i in range(len(MLPweights)):
            torch.manual_seed(self.init_base_seed+i+1)
            init_func(tensor=MLPweights[i].T, **init_kwargs)
        
        if self.lora_rank > 0:
            LoRA_weights = [self.lora_A, self.lora_B]
            for i in range(len(LoRA_weights)):
                torch.manual_seed(self.lora_init_base_seed+i+1)
                init_func(tensor=LoRA_weights[i].T, **init_kwargs)
                    
            
    
class SparseMLPWithLoRA(nn.Module):
    """Sparse MLP module with LoRA adapters
    This is a GLU-style sparse MLP layer with LoRA adapters, \
        where the sparcity is implemented as Mixture of Experts (MoE), \
            and each expert is a dense MLP with LoRA adapters.
    """
    
    def __init__(self,
        hidden_size: int,
        ffh_size: int,
        activation_type: MLPActivationType = MLPActivationType.SILU,
        num_experts: int = 1,
        moe_topk: int = 1,
        rank: int = 0,
        world_size: int = 1,
        process_group: Optional[ProcessGroup] = None,
        init_mean: float = 0.0,
        init_std: float = 1.0,
        init_base_seed: int = 42,
        lora_rank: int = 0,
        lora_alpha: Optional[float] = None,
        lora_dropout_rate: float = 0.0,
        lora_dropout_seed: int = 42,
        lora_init_base_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """Initialize Sparse MLP module with LoRA adapters
        
        Args:
            hidden_size(int): hidden dimension size
            ffh_size(int): hidden dimension size
            activation_type(MLPActivationType, default = MLPActivationType.SILU): activation type
            num_experts(int, default = 1): number of (global) experts, which can deduce expert_size = ffh_size // num_experts
            moe_topk(int, default = 1): topk-routing for MoE to control the sparcity
            rank(int, default = 0): rank
            world_size(int, default = 1): world size
            process_group(Optional[ProcessGroup], default = None): the process group (which will not be used for this simpler module yet)
            init_mean(float, default = 0.0): mean for the initialization
            init_std(float, default = 1.0): std for the initialization
            init_base_seed(int, default = 42): seed for the initialization
            lora_rank(int, default = 0): lora rank
            lora_alpha(Optional[float], default = None): lora alpha
            lora_dropout_rate(float, default = 0.0): lora dropout rate
            lora_dropout_seed(int, default = 42): lora dropout seed
            lora_init_base_seed(int, default = 42): seed for lora weight initialization
            dtype(torch.dtype, default = torch.float32): parameter dtype
            device(str, default = "cpu"): parameter device
        """
        super().__init__()
        kwargs = {"device": device, "dtype": dtype}
        self.hidden_size = hidden_size
        self.ffh_size = ffh_size
        self.num_experts = num_experts
        self.ffh_per_expert = hidden_size // num_experts
        self.moe_topk = moe_topk
        self.rank = rank
        self.world_size = world_size
        self.num_local_experts = num_experts // world_size
        self.init_mean = init_mean
        self.init_std = init_std
        self.init_base_seed = init_base_seed
        # G linear weights:
        self.G = nn.Parameter(torch.empty(hidden_size, num_experts, **kwargs))
        # Mix of Experts Architecture with 'nle' local experts
        self.experts = nn.ModuleList(
            [DenseMLPWithLoRA(
                hidden_size=self.hidden_size,
                ffh_size=self.ffh_per_expert,
                activation_type=activation_type,
                init_base_seed=init_base_seed+i+rank*self.num_local_experts,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout_rate=lora_dropout_rate,
                lora_dropout_seed=lora_dropout_seed+i+rank*self.num_local_experts,
                lora_init_base_seed=lora_init_base_seed+i+rank*self.num_local_experts,
                dtype=dtype,
                device=device
            ) for i in range(self.num_local_experts)])
        
        self.reset_parameters()
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward pass of the Sparse MLP module with LoRA adapters
        
        Args:
            input(torch.Tensor): input tensor, with shape: [batch_size, seq_len, hidden_size]
            
        Returns:
            output(torch.Tensor): output tensor, with shape: [batch_size, seq_len, hidden_size]
        """
        b,s,h = input.shape
        input_dtype = input.dtype
        output_state = torch.zeros_like(input, dtype=torch.float32, device=input.device)
        # for token t :
        # Pt = softmax(Xt G)
        router_score = F.softmax(F.linear(input.to(torch.float32), self.G.T), dim=-1, dtype=torch.float32)
        # It = arg-topk(Pt), Qt = Pt[It]
        value, indices = torch.topk(router_score, self.moe_topk, dim=-1)
        # Wt = Qt/sum(Qt)
        experts_weights = value / torch.sum(value, dim=-1, keepdim=True)
        # transfer to 'one-hot' encode
        # mask is with shape [b,s,k,ne]
        experts_mask = F.one_hot(indices, num_classes=self.num_experts)
        
        # permute to shape [ne, k, s, b]
        # ne : idx of this expert
        # k : weights of this expert is at top<k>
        # s, b: weights of this expert at top<k> corresponds to input[b, s]
        experts_mask = experts_mask.permute(3, 2, 1, 0)
        
        # Loop over the local experts to get the correct weight
        offset_base = self.rank * self.num_local_experts
        for expert_idx in range(self.num_local_experts):
            expert_layer = self.experts[expert_idx]
            top_idx, s_idx, b_idx = torch.where(experts_mask[expert_idx+offset_base])
            
            # get the weights tensor for expert <expert_idx>
            current_weight = torch.zeros((b,s), dtype=torch.float32, device=input.device)
            current_weight[b_idx, s_idx] = value[b_idx, s_idx, top_idx]
            
            # get hidden_state return from expert <expert_idx>:
            current_hidden_state = expert_layer(input) * current_weight.unsqueeze(dim=-1)
            
            output_state = output_state + current_hidden_state
        
        return output_state.to(input_dtype)
        
    def reset_parameters(self):
        """Initialize the weights of each local expert from its own distribution \
            and the gating layer from a normal distribution
        """
        torch.manual_seed(self.init_base_seed)
        nn.init.normal_(self.G, self.init_mean, self.init_std)

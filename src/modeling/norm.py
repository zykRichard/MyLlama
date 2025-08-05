import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupRMSNorm(nn.Module):
    """Group RMS Norm module
    This is a variant of RMS Norm that \
        evenly splits the hidden dimension into groups, and \
        applies root-mean-square normalization with \
            learnable scaling transformation on each i-th group individually.
    """
    
    def __init__(self, 
        hidden_size: int, 
        group_size: int,
        eps: float = 1e-5,
        init_range: tuple = (-1.0, 1.0),
        init_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        """Initialize Group RMS Norm module
        
        Args:
            hidden_size(int): hidden dimension size
            group_size(int): group size
            eps(float, default = 1e-5): epsilon
            init_range(tuple, default = (-1.0, 1.0)): the range of the uniform distribution to initialize learnable scaling parameters
            init_seed(int, default = 42): seed for the initialization
            dtype(torch.dtype, default = torch.float32): parameter dtype
            device(str, default = "cpu"): parameter device
        """
        super().__init__()
        self.weights = nn.Parameter(torch.empty(hidden_size, dtype=dtype)).to(device)
        self.gz = group_size
        self.eps = eps
        self.init_range = init_range
        self.init_seed = init_seed
        self.dtype = dtype
        self.device = device
        
        if hidden_size % group_size != 0:
            raise ValueError("RMSNorm : HIDDEN_SIZE isn't divisible by GROUP_SIZE !")
        
        self.gd = hidden_size // group_size
        self.reset_parameters() 
        
    def forward(self, input : torch.Tensor) -> torch.Tensor:
        """The forward pass for Group RMS Norm module

        Args:
            input(torch.Tensor): input tensor, with shape: [batch_size, seq_len, hidden_size]
            
        Returns:
            output(torch.Tensor): normalized output tensor, with shape: [batch_size, seq_len, hidden_size]
        """
        b, s = input.shape[0:2]
        input_dtype = input.dtype
        input_group = input.reshape(b, s, self.gz, self.gd) #[b,s,gz,gd]
        
        # to ensure the high precise, use fp32 type
        input_group = input_group.to(torch.float32)
        variance = input_group.pow(2).mean(-1, keepdim=True)
        input_group = input_group * torch.rsqrt(variance + self.eps)
        
        hidden_state = input_group * self.weights.reshape(self.gz, self.gd)
        hidden_state = hidden_state.reshape(b, s, -1)
        return hidden_state.to(dtype=input_dtype)
    
    def reset_parameters(self) -> None:
        """Initialize learnable scaling parameters for Group RMS Norm from a uniform distribution"""
        torch.manual_seed(self.init_seed)
        nn.init.uniform_(self.weights, self.init_range[0], self.init_range[1])


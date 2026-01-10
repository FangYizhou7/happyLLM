from transformers import PretrainedConfig
import torch
import torch.nn as nn
class ModelConfig(PretrainedConfig):
    model_type = "Tiny-k"
    def __init__(self,
                 dim:int=768,
                 n_layers:int=12,
                 n_heads:int=16,
                 n_kv_heads:int=8,
                 vocab_size:int=6144,
                 hidden_dim:int=None,
                 multiple_of:int=64,
                 norm_eps:float=1e-5,
                 max_seq_len:int=512,
                 dropout:float=0.0,
                 flash_attn:bool=True,
                 **kwargs,):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)


class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float):
        super().__init__()
        self.eps=eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self,x):
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
    def forward(self,x):
        output=self._norm(x.float()).type_as(x)
        return output*self.weight
from typing import Optional

import torch
from torch import nn
from transformers import PreTrainedModel

from config.ModelConfig import ModelConfig, RMSNorm
from model.attention import precompute_freqs_cis
from model.decoderLayer import DecoderLayer


class Transformer(PreTrainedModel):
    config_class=ModelConfig
    last_loss:Optional[torch.Tensor]

    def __init__(self, args:ModelConfig=None):
        super().__init__(args)

        #初始化模型参数
        self.args=args

        #词汇表大小
        self.vocab_size=args.vocab_size

        #层数
        self.n_layers=args.n_layers

        #词嵌入层
        self.tok_embeddings=nn.Embedding(args.vocab_size, args.dim)

        #dropout层
        self.dropout=nn.Dropout(args.dropout)
        #Decoder层
        self.layers=torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(DecoderLayer(layer_id, args))

        #归一化层
        self.norm=RMSNorm(args.dim,eps=args.norm_eps)

        #输出层
        self.output=nn.Linear(args.dim,args.vocab_size,bias=False)

        #将词嵌入层的权重和输出层的权重共享
        self.tok_embeddings.weight=self.output.weight

        #预计算相对位置嵌入的概率
        freqs_cos,freqs_sin=precompute_freqs_cis(self.args.dim//self.args.n_heads,self.args.max_seq_len)

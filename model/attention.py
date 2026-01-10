from typing import Tuple

import torch
from torch import nn

from config.ModelConfig import ModelConfig


class Attention(nn.Module):
    def __init__(self,args:ModelConfig):
        super().__init__()
        self.n_kv_heads=args.n_heads if args.n_kv_heads is None else args.n_kv_heads

        #确保总头数可以被键值头数整除
        assert args.n_heads % args.n_kv_heads == 0

        #模型并行处理大小，默认为1
        model_parallel_size=1
        # 本地计算头数 等于总头数除以模型并行处理大小
        self.n_local_heads=args.n_heads // model_parallel_size

        #本地键值头数 等于总键值头数除以模型并行处理大小
        self.n_local_kv_heads=self.n_kv_heads // model_parallel_size

        #重复次数 用于扩展键和值的尺寸
        self.n_rep=self.n_local_heads //  self.n_local_kv_heads

        #每个头的维度 等于模型维度除以头的总数
        self.head_dim=args.dim// args.n_heads


        #定义权重矩阵
        self.wq=nn.Linear(args.dim,args.n_heads*self.head_dim,bias=False)
        self.wk=nn.Linear(args.dim,self.n_kv_heads*self.head_dim,bias=False)
        self.wv=nn.Linear(args.dim,self.n_kv_heads*self.head_dim,bias=False)

        #输出权重矩阵
        self.wo=nn.Linear(args.n_heads*self.head_dim,args.dim,bias=False)

        #定义dropout
        self.attn_dropout=nn.Dropout(args.dropout)
        self.resid_dropout=nn.Dropout(args.dropout)





# 实现匹配多头注意力
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # 获取输入张量的形状：批量大小、序列长度、键/值对头的数量、每个头的维度大小
    bs, slen, n_kv_heads, head_dim = x.shape

    # 如果重复次数为1，则不需要重复，直接返回原始张量
    if n_rep == 1:
        return x

    # 对张量进行扩展和重塑操作以重复键值对
    return (
        x[:, :, :, None, :]  # 在第四个维度（头的维度前）添加一个新的维度
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)  # 将新添加的维度扩展到n_rep大小，实现重复的效果
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)  # 重新塑形，合并键/值对头的数量和重复次数的维度
    )




def precompute_freqs_cis(dim:int,end:int,theta:float=10000.0):
    freqs=1.0/(theta**(torch.arange(0,dim,2)[:(dim//2)].float()/dim))
    t=torch.arange(end,device=freqs.device)

    freqs=torch.outer(t,freqs).float()
    freqs_cos=torch.cos(freqs)

    freqs_sin=torch.sin(freqs)

    return freqs_cos,freqs_sin


def reshape_for_broadcast(freqs_cis:torch.Tensor,x:torch.Tensor):
    # 获取x的维度
    ndim=x.ndim

    # 断言确保1在x的维度范围内
    assert 0<=1<ndim

    assert freqs_cis.shape==(x.shape[1],x.shape[-1])

    # 构造一个新的形状，除了第二维和最后一维，其他维度都为1，这样做是为了能够将freqs_cis与x进行广播操作
    shape=[d if i==1 or i==ndim-1 else 1 for i,d in enumerate(x.shape)]

    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # 将查询和键张量转换为浮点数，并重塑形状以分离实部和虚部
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # 重新塑形频率张量以进行广播
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # 应用旋转，分别计算旋转后的实部和虚部
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # 将最后两个维度合并，并还原为原始张量的形状
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


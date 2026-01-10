import torch

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


# 实现旋转嵌入
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

import math


def get_lr(it,all):
    """
    计算当前迭代的学习率，使用余弦退火调度策略

    学习率调度策略：
    1. Warmup阶段：学习率从0线性增长到目标学习率
    2. 余弦退火阶段：学习率按余弦函数衰减到最小学习率
    3. 超出训练步数后：保持最小学习率

    Args:
        it (int): 当前迭代步数
        all (int): 总迭代步数

    Returns:
        float: 当前步数对应的学习率
    """
    warmup_iters=args.warmup_iters  #预热迭代次数
    lr_decay_iters = all  #学习率衰减的总迭代次数
    min_lr = args.learning_rate  / 10  # 最小学习率,为初始学习率的 1/10

    # warmup阶段:线性增长
    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters

    if it > lr_decay_iters:
        return min_lr
    #余弦退火阶段
    decay_ratio=(it-warmup_iters)/ (lr_decay_iters-warmup_iters)
    assert 0<=decay_ratio<=1
    coeff=0.5*(1.0+math.cos(math.pi*decay_ratio)) #余弦系数
    return min_lr+coeff*(args.learning_rate - min_lr)
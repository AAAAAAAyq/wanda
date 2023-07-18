import math
import time

import torch
import torch.nn as nn
import transformers

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

## SparseGPT: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
class SparseGPT:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev) # (in_dim, in_dim)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()   # (in_dim, bs*seqlen)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())


    def fasterprune(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)    # 每个out_dim的Loss

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp   # 对角线增加自适应damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None
        # (rows=output, columns=input)
        for i1 in range(0, self.columns, blocksize):    # 分块对列进行剪枝(group input)
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()    # (rows, blocksize)
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1) # TODO:Err1和Losses1的区别是什么?
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]  # Hessian逆的子块 (blocksize, blocksize)

            if prune_n == 0:    # 非结构化剪枝
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2   # (rows, blocksize) TODO:为什么要平方?[cholesky分解导致的]
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]  # 一组输入内的r%=整体r%
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]    # (rows, 1)
                d = Hinv1[i, i] # (1,)

                if prune_n != 0 and i % prune_m == 0:   # 在blocksize中按照N:M分块
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0  # 掩码后的w

                Q1[:, i] = q    # 掩码后的W1
                Losses1[:, i] = (w - q) ** 2 / d ** 2   # OBS Loss (只计算剪去的权重)

                err1 = (w - q) / d  # 权重补偿的中间量 (只计算剪去的权重)
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))    # blocksize内部权重补偿公式
                Err1[:, i] = err1   # 用于更新后续blocksize的权重

            W[:, i1:i2] = Q1    # blocksize的剪枝后权重
            Losses += torch.sum(Losses1, 1) / 2 # 每个output特征的OBS loss[补偿后的L2损失?]

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])  # 给后续blocksize更新权重

        torch.cuda.synchronize()
        print(f"Losses: {Losses.mean()}")
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()
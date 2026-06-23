import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ultralytics.nn.modules.conv import Conv as UltraConv  



class BaseConv(nn.Module):
    """Standard convolution with BN and activation, internally used by CDC."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p if p is not None else k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))



class RepConv_CDC(nn.Module):


    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, act=True, bn=False, theta=0.5):
        super().__init__()
        assert k == 3 and p == 1, "CDC must be 3x3 convolution"
        self.g = g
        self.c1 = c1
        self.c2 = c2


        self.theta = nn.Parameter(torch.tensor(theta, dtype=torch.float32))
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.conv1 = BaseConv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = BaseConv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)
        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None

    def forward(self, x):
        out_3x3 = self.conv1.conv(x)
        out_3x3 = self.conv1.bn(out_3x3)

        if self.theta != 0:
            w = self.conv1.conv.weight
            w_sum = w.sum(dim=(2, 3), keepdim=True)
            out_diff = F.conv2d(x, w_sum, stride=self.conv1.conv.stride, groups=self.g)
            out_diff = self.conv1.bn(out_diff)
            out_3x3 = out_3x3 - self.theta * out_diff

        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(out_3x3 + self.conv2(x) + id_out)

    def fuse_convs(self):
        if hasattr(self, 'conv'):
            return

        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)

        if self.theta != 0:
            w_sum = kernel3x3.sum(dim=(2, 3), keepdim=True)
            cdc_correction = torch.zeros_like(kernel3x3)
            cdc_correction[:, :, 1:2, 1:2] = w_sum * self.theta
            kernel3x3 = kernel3x3 - cdc_correction

        fused_kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid
        fused_bias = bias3x3 + bias1x1 + biasid

        self.conv = nn.Conv2d(in_channels=self.c1, out_channels=self.c2,
                              kernel_size=3, stride=self.conv1.conv.stride,
                              padding=1, groups=self.g, bias=True).requires_grad_(False)
        self.conv.weight.data = fused_kernel
        self.conv.bias.data = fused_bias

        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'bn'): self.__delattr__('bn')

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        return F.pad(kernel1x1, [1, 1, 1, 1]) if kernel1x1 is not None else 0

    def _fuse_bn_tensor(self, branch):
        if branch is None: return 0, 0
        if isinstance(branch, BaseConv):
            kernel = branch.conv.weight
            running_mean, running_var = branch.bn.running_mean, branch.bn.running_var
            gamma, beta, eps = branch.bn.weight, branch.bn.bias, branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            input_dim = self.c1 // self.g
            kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.c1): kernel_value[i, i % input_dim, 1, 1] = 1
            kernel = torch.from_numpy(kernel_value).to(branch.weight.device)
            running_mean, running_var = branch.running_mean, branch.running_var
            gamma, beta, eps = branch.weight, branch.bias, branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std



class GA_RepCDC(nn.Module):


    def __init__(self, c1, c2, n=3, e=1.0, theta=0.1):  
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = BaseConv(c1, c_, 1, 1)
        self.cv2 = BaseConv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv_CDC(c_, c_, theta=theta) for _ in range(n)])
        self.cv3 = BaseConv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))
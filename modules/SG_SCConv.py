import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

__all__ = ['SCConvBasicBlock3']


# ==================== 1. 鲁棒版 SRU (Soft Spatial Reconstruction Unit) ====================
class SRU(nn.Module):
    def __init__(self, oup_channels, group_num=8):
        super().__init__()
        # 1. 减小 group_num：红外目标小，Group 太大容易平滑掉特征。
        # 2. affine=True 是默认的，这里显式写出来强调我们利用它
        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num, affine=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()

        # 1. 计算 GroupNorm
        # gn_x = (x - mean) / std * gamma + beta
        # 这里包含了最核心的信息：
        # - (x-mean)/std: 突出了相对于背景的异常点 (High Frequency)
        # - gamma: 学习该通道的重要性
        # - beta: 学习该通道的激活阈值
        gn_x = self.gn(x)

        # 2. 【核心修正】直接生成门控，移除所有人为的权重归一化
        # 以前：reweights = sigmoid(gn_x * normalized_weight) -> 也就是通道间有竞争
        # 现在：reweights = sigmoid(gn_x) -> 通道独立，只看当前特征够不够“异常”

        # 为什么这样对 CDC 最好？
        # 如果 gn_x 某处值很大（发现了小目标），sigmoid 直接输出 ~1.0。
        # 不管其他通道发生了什么，这个高频信号都会被完整保留。
        reweights = self.sigmoid(gn_x)

        # 3. 软加权分离 (Soft Split)
        # 这一步保持不变，确保梯度流平滑
        x_1 = x * reweights  # Info (目标/高频)
        x_2 = x * (1 - reweights)  # Noise (背景/低频) -> 留给 CDC 做差分参考

        return torch.cat([x_1, x_2], dim=1)

# ==================== 2. 鲁棒版 CRU (Channel Reconstruction Unit) ====================
class CRU(nn.Module):
    def __init__(self, op_channel, alpha=1 / 2, squeeze_radio=2, group_size=2):
        super().__init__()
        # op_channel 是 SRU 输出的 (2 * C)

        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel

        # 这里的 squeeze 输出通道为了还原
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)

        # 目标输出通道数 (还原回原始输入通道数 C)
        out_ch = op_channel // 2

        # 鲁棒性检查：确保分组卷积能够整除
        mid_ch = up_channel // squeeze_radio
        if mid_ch % group_size != 0:
            # 如果不能整除，强制组数为 1 (普通卷积)，防止报错
            group_size = 1

        # GWC: Group-wise Convolution (处理高频/目标信息)
        self.GWC = nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=group_size, bias=False)

        # PWC: Point-wise Convolution (处理低频/背景信息)
        self.PWC1 = nn.Conv2d(low_channel // squeeze_radio, out_ch, kernel_size=1, bias=False)

        # 【核心修正】：可学习的融合参数
        # 初始化为 0.0，经过 Sigmoid 后变为 0.5 (初始时同等看待)
        self.alpha_param = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: [B, 2C, H, W]
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)

        # Squeeze
        up = self.squeeze1(up)
        low = self.squeeze2(low)

        # Transform
        Y1 = self.GWC(up)  # Info branch
        Y2 = self.PWC1(low)  # Context branch

        # 【核心修正】：使用 Sigmoid 约束参数范围在 [0, 1]
        # 这样保证了数值稳定性，防止梯度爆炸
        fusion_ratio = torch.sigmoid(self.alpha_param)

        # 加权融合
        out = Y1 * fusion_ratio + Y2 * (1 - fusion_ratio)

        return out


# ==================== 3. SCConv 封装 ====================
class SCConv(nn.Module):
    def __init__(self, oup_channel):
        super().__init__()
        # 通道数检查
        assert oup_channel % 2 == 0, "SCConv channel must be divisible by 2"
        self.sru = SRU(oup_channel)
        self.cru = CRU(oup_channel * 2)

    def forward(self, x):
        x = self.sru(x)
        x = self.cru(x)
        return x


# ==================== 4. SCConvBasicBlock (RT-DETR 插件) ====================
class SCConvBasicBlock3(nn.Module):
    expansion = 1
    # 这一行必须加，否则 Ultralytics 解析 yaml 可能报错
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__()
        self.shortcut = shortcut

        # 1. 正常的卷积层，调整通道
        self.conv1 = Conv(ch_in, ch_out, 3, stride, act=act)

        # 2. SCConv 模块 (特征提纯)
        # 输入输出通道必须一致
        self.scconv = SCConv(ch_out)

        # 3. Shortcut 分支处理
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(
                    nn.AvgPool2d(2, 2, 0, ceil_mode=True),
                    Conv(ch_in, ch_out, 1, 1)
                )
            else:
                self.short = Conv(ch_in, ch_out, 1, stride)

        self.act = nn.Identity() if act is None else (nn.ReLU(inplace=True) if act == 'relu' else nn.SiLU(inplace=True))

    def forward(self, x):
        # 卷积
        y = self.conv1(x)

        # 软加权空间重建 (Soft-SCConv)
        # 这里 y 的梯度可以顺滑地传回 conv1
        y = self.scconv(y)

        # 残差连接
        if self.shortcut:
            out = x + y
        else:
            out = self.short(x) + y

        return self.act(out)
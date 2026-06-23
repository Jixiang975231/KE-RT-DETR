import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class MultiDirConv(nn.Module):
    def __init__(self):
        super(MultiDirConv, self).__init__()
        k1 = [[-0.5, 0, 0], [0, 1, 0], [0, 0, -0.5]]
        k2 = [[0, -0.5, 0], [0, 1, 0], [0, -0.5, 0]]
        k3 = [[0, 0, -0.5], [0, 1, 0], [-0.5, 0, 0]]
        k4 = [[0, 0, 0], [-0.5, 1, -0.5], [0, 0, 0]]
        kernels = torch.FloatTensor([k1, k2, k3, k4]).unsqueeze(1)
        self.weight = nn.Parameter(data=kernels, requires_grad=False)

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=1)


class ECA(nn.Module):


    def __init__(self, kernel_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class OptimizedMixedFusion(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OptimizedMixedFusion, self).__init__()


        self.mix_conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.mix_bn = nn.BatchNorm2d(out_ch)
        self.mix_act = nn.SiLU(inplace=True) 


        self.spatial_conv = nn.Conv2d(out_ch, out_ch, kernel_size=5, padding=2, groups=out_ch, bias=False)
        self.spatial_bn = nn.BatchNorm2d(out_ch)
        self.spatial_act = nn.SiLU(inplace=True)


        self.attn = ECA(kernel_size=3)

    def forward(self, x_concat):

        x = self.mix_conv(x_concat)
        x = self.mix_bn(x)
        x = self.mix_act(x)


        x = self.spatial_conv(x)
        x = self.spatial_bn(x)
        x = self.spatial_act(x)


        x = self.attn(x)

        return x



class SPCE(nn.Module):
    def __init__(self, c1, c2, *args):
        super(SPCE, self).__init__()


        self.channel_adapter = nn.Conv2d(c1, 1, 1, bias=False) if c1 != 1 else nn.Identity()
        self.dir_conv = MultiDirConv()


        self.stem = nn.Sequential(
            nn.BatchNorm2d(1, affine=False),
            nn.Conv2d(1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )


        self.fusion = OptimizedMixedFusion(in_ch=c2 + 5, out_ch=c2)

    def forward(self, x):

        x_gray = self.channel_adapter(x)
        if x_gray.size(1) != 1: x_gray = x_gray.mean(dim=1, keepdim=True)

        dir_feats = F.relu(self.dir_conv(x_gray))

        eps = 1e-6
        f_mean = torch.mean(dir_feats, dim=1, keepdim=True)
        f_min, _ = torch.min(dir_feats, dim=1, keepdim=True)
        f_std = torch.std(dir_feats, dim=1, keepdim=True)
        d1, d2, d3, d4 = torch.split(dir_feats, 1, dim=1)
        f_cross = d1 * d3 + d2 * d4
        f_harmonic = 4.0 / (torch.sum(1.0 / (dir_feats + eps), dim=1, keepdim=True))
        f_snr = f_mean / (f_std + eps)

        stat_feats = torch.cat([f_cross, f_mean, f_min, f_harmonic, f_snr], dim=1)

        deep_feats = self.stem(x_gray)


        combined = torch.cat([deep_feats, stat_feats], dim=1)


        out = self.fusion(combined)

        return out


if __name__ == "__main__":
    c1, c2 = 3, 32
    model = SPCE(c1, c2)
    x = torch.randn(1, c1, 640, 640)
    out = model(x)

    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")


    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Params: {total_params}")
    print("Optimization Checklist:")
    print("1. [OK] 5x5 Depthwise Kernel for better context")
    print("2. [OK] SiLU Activation for better gradients")
    print("3. [OK] ECA Attention for dynamic feature selection")
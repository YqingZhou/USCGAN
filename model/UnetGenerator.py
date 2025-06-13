"""
不带CBAM注意力机制的U-Net生成器模型（用于CycleGAN）

基于原始代码修改
移除所有CBAM (卷积块注意力模块)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """卷积块：用于下采样或上采样操作"""
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """没有CBAM的UNet生成器"""
    def __init__(self, img_channels, num_features=64, num_residuals=2):
        super().__init__()

        # 初始卷积块
        self.initial = nn.Sequential(
            nn.Conv2d(
                img_channels,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )

        # 编码器（下采样）块
        self.down1 = ConvBlock(
            num_features, num_features * 2, kernel_size=3, stride=2, padding=1
        )
        self.down2 = ConvBlock(
            num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1
        )
        self.down3 = ConvBlock(
            num_features * 4, num_features * 8, kernel_size=3, stride=2, padding=1
        )

        # 瓶颈中的残差块
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 8) for _ in range(num_residuals)]
        )

        # 解码器（上采样）块
        self.up1 = ConvBlock(
            num_features * 8,
            num_features * 4,
            down=False,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.up2 = ConvBlock(
            num_features * 8,  # *4 + *4 due to skip connection
            num_features * 2,
            down=False,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.up3 = ConvBlock(
            num_features * 4,  # *2 + *2 due to skip connection
            num_features,
            down=False,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

        # 最终输出层
        self.final = nn.Conv2d(
            num_features * 2,  # *1 + *1 due to skip connection
            img_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

    def forward(self, x):
        # 初始块
        x1 = self.initial(x)  # [B, 64, H, W]

        # 编码器
        x2 = self.down1(x1)   # [B, 128, H/2, W/2]
        x3 = self.down2(x2)   # [B, 256, H/4, W/4]
        x4 = self.down3(x3)   # [B, 512, H/8, W/8]

        # 瓶颈
        x4 = self.res_blocks(x4)

        # 解码器带跳跃连接
        x = self.up1(x4)             # [B, 256, H/4, W/4]
        x = torch.cat([x, x3], dim=1)  # 直接使用x3，不需要注意力

        x = self.up2(x)             # [B, 128, H/2, W/2]
        x = torch.cat([x, x2], dim=1)  # 直接使用x2，不需要注意力

        x = self.up3(x)             # [B, 64, H, W]
        x = torch.cat([x, x1], dim=1)  # 直接使用x1，不需要注意力

        # 最终输出带tanh激活
        return torch.tanh(self.final(x))


def test():
    """测试函数：检查模型形状和参数量"""
    img_channels = 3
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = Generator(img_channels, num_residuals=2)
    print(f"输出形状: {gen(x).shape}")

    # 计算模型参数
    num_params = sum(p.numel() for p in gen.parameters())
    print(f"参数数量: {num_params:,}")

    # 计算FLOPS（粗略估计）
    def count_conv2d(m, x, y):
        x = x[0]
        cin = m.in_channels
        cout = m.out_channels
        kh, kw = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
        batch_size = x.size()[0]
        out_h = y.size(2)
        out_w = y.size(3)

        # ops per output element
        kernel_ops = kh * kw * cin
        bias_ops = 1 if m.bias is not None else 0
        ops_per_element = kernel_ops + bias_ops

        # total ops
        output_elements = batch_size * out_h * out_w * cout
        total_ops = output_elements * ops_per_element

        print(f"{m}: FLOPs: {total_ops / 1e9:.3f}G")

    # 例如估算第一层的FLOPs
    first_conv = gen.initial[0]
    fake_input = torch.zeros((1, img_channels, img_size, img_size))
    fake_output = first_conv(fake_input)
    count_conv2d(first_conv, [fake_input], fake_output)


if __name__ == "__main__":
    test()
"""
Discriminator model for CycleGAN with Self-Attention mechanism and Spectral Normalization
Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
* 2020-11-05: Initial coding
* 2022-12-21: Small revision of code, checked that it works with latest PyTorch version
* Modified to include self-attention mechanism only in 256 channel layer
* Modified to include spectral normalization on all layers except the output layer
* Modified to remove InstanceNorm from all layers
"""
import torch
import torch.nn as nn


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Dimensions reduction factor for attention calculations (typical value is 8)
        self.channels = in_channels
        self.query = SpectralNorm(nn.Conv2d(in_channels, in_channels // 8, kernel_size=1))
        self.key = SpectralNorm(nn.Conv2d(in_channels, in_channels // 8, kernel_size=1))
        self.value = SpectralNorm(nn.Conv2d(in_channels, in_channels, kernel_size=1))
        self.gamma = nn.Parameter(torch.zeros(1) * 0.1)  # Initialize with small value for smoother training

    def forward(self, x):
        # Input shape: [batch_size, channels, height, width]
        batch_size, C, height, width = x.size()

        # Get query, key, value projections
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B x (H*W) x (C//8)
        key = self.key(x).view(batch_size, -1, height * width)  # B x (C//8) x (H*W)
        value = self.value(x).view(batch_size, -1, height * width)  # B x C x (H*W)

        # Calculate attention scores (matrix multiplication of query and key)
        attention = torch.bmm(query, key)  # B x (H*W) x (H*W)
        attention = torch.softmax(attention, dim=-1)  # Normalize attention scores

        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x (H*W)
        out = out.view(batch_size, C, height, width)  # Reshape back to original shape

        # Apply gamma scaling factor and add to input as residual connection
        out = self.gamma * out + x

        return out


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_attention=False):
        super().__init__()
        self.use_attention = use_attention

        self.conv = nn.Sequential(
            SpectralNorm(nn.Conv2d(
                in_channels,
                out_channels,
                4,
                stride,
                1,
                bias=True,
                padding_mode="reflect",
            )),
            # InstanceNorm2d 已被移除
            nn.LeakyReLU(0.2, inplace=True),
        )

        if use_attention:
            self.attention = SelfAttention(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_attention:
            x = self.attention(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            SpectralNorm(nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            )),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels = features[0]

        # Only apply self-attention on the third layer (256 channels)
        for idx, feature in enumerate(features[1:]):
            # Only apply self attention to the 256 channel layer (idx=1)
            use_attention = (idx == 1)

            layers.append(
                Block(
                    in_channels,
                    feature,
                    stride=1 if feature == features[-1] else 2,
                    use_attention=use_attention
                )
            )
            in_channels = feature

        # 注意：最后一层输出层不应用谱归一化
        layers.append(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))


def test():
    x = torch.randn((5, 3, 256, 256))
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(preds.shape)

    # Calculate number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")


if __name__ == "__main__":
    test()
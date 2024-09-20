import torch
import torch.nn as nn
import torch.nn.functional as F

class FCA(nn.Module):
    def __init__(self, inp, oup, mode="tf"):
        super(FCA, self).__init__()

        layers = [
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
        ]

        mode_weights = {
            "temp": [
                nn.Conv2d(oup, oup, kernel_size=(1, 5), padding=(0, 2), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(1, 5), padding=(0, 2), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            ],
            "freq": [
                nn.Conv2d(oup, oup, kernel_size=(5, 1), padding=(2, 0), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5, 1), padding=(2, 0), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            ],
            "tf": [
                nn.Conv2d(oup, oup, kernel_size=(1, 5), padding=(0, 2), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5, 1), padding=(2, 0), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            ],
        }

        layers.extend(mode_weights.get(mode, []))

        layers.append(nn.Sigmoid())

        self.attn = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, C, F, T)
        attn_map = F.interpolate(self.attn(x), size=(x.shape[-2], x.shape[-1]), mode='nearest')
        return attn_map

class Sandglass(nn.Module):
    def __init__(self, inp, oup, mid, ksize=3, stride=1):
        super(Sandglass, self).__init__()
        
        # First depthwise convolution
        self.dw1 = nn.Sequential(
            nn.Conv2d(inp, inp, kernel_size=ksize, stride=stride, padding=ksize // 2, bias=False, groups=inp),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
        )
        
        # Pointwise convolution (reduce)
        self.pw_reduce = nn.Sequential(
            nn.Conv2d(inp, mid, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid),
        )
        
        # Pointwise convolution (expand)
        self.pw_expand = nn.Sequential(
            nn.Conv2d(mid, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),
        )
        
        # Second depthwise convolution
        self.dw2 = nn.Sequential(
            nn.Conv2d(oup, oup, kernel_size=ksize, stride=stride, padding=ksize // 2, bias=False, groups=oup),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),
        )

        self.use_residual = (inp == oup)
        
    def forward(self, x):
        out = self.dw1(x)
        out = self.pw_reduce(out)
        out = self.pw_expand(out)
        out = self.dw2(out)
        if self.use_residual:
            out += x
        return out
        
class FCABlock(nn.Module):
    def __init__(self, inp, oup, mid_channels=None, ksize=3, stride=1, at_mode="tf"):
        super(FCABlock, self).__init__()
        assert stride in [1, 2], "Stride must be 1 or 2"
        mid = mid_channels if mid_channels is not None else oup // 2

        # Attention module
        self.attn = FCA(inp, oup, mode=at_mode)

        self.res1 = nn.Sequential(
            nn.Conv2d(inp, mid, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(mid, oup - mid, kernel_size=ksize, stride=1, padding=ksize // 2, groups=mid, bias=False),
            nn.BatchNorm2d(oup - mid),
            nn.ReLU6(inplace=True),
        )

        self.FF = nn.Sequential(
            nn.Conv2d(oup, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),
        )

        # Shortcut connection
        if stride == 1 and inp == oup:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=ksize, stride=stride, padding=ksize // 2, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        r1 = self.res1(x)
        r2 = self.res2(r1)
        res = torch.cat((r1, r2), dim=1)
        attn_map = self.attn(x)
        out = self.FF(attn_map * res) + self.shortcut(x)
        return out



class lmfcaNet(nn.Module):
    def __init__(self, in_ch=6, out_ch=2):
        super(lmfcaNet, self).__init__()

        channels = [48, 96, 224, 480]

        # First Block
        self.firstblock = nn.Sequential(
            FCABlock(inp=in_ch, oup=channels[0]),
            Sandglass(channels[0], channels[0], channels[0] // 2, ksize=3, stride=1),
        )

        # Encoder (Downsampling) Layers
        self.maxpools = nn.ModuleList([nn.MaxPool2d(2, 2) for _ in range(4)])
        at_modes = ["freq", "temp", "freq"]
        self.down_blocks = nn.ModuleList()
        for idx in range(3):
            self.down_blocks.append(nn.Sequential(
                FCABlock(channels[idx], channels[idx + 1], ksize=3, stride=1, at_mode=at_modes[idx]),
                Sandglass(channels[idx + 1], channels[idx + 1], channels[idx], ksize=3, stride=1),
            ))

        # Decoder (Upsampling) Layers
        self.up_blocks = nn.ModuleList()
        
        # Up4
        self.up_blocks.append(nn.Sequential(
            Sandglass(channels[3], channels[3], channels[2], ksize=3, stride=1),
            Sandglass(channels[3], channels[3], channels[2], ksize=3, stride=1),
            nn.ConvTranspose2d(channels[3], channels[3], kernel_size=2, stride=2, padding=0),
        ))
        # Up3 to Up1
        for idx in range(2, -1, -1):
            self.up_blocks.append(nn.Sequential(
                FCABlock(channels[idx + 1], channels[idx]),
                Sandglass(channels[idx], channels[idx], channels[max(idx - 1, 0)], ksize=3, stride=1),
                nn.ConvTranspose2d(channels[idx], channels[idx], kernel_size=2, stride=2, padding=0),
            ))

        self.lastConv = nn.Sequential(
            Sandglass(channels[0], channels[0], channels[0] // 2, ksize=3, stride=1),
            Sandglass(channels[0], channels[0], channels[0] // 2, ksize=3, stride=1),
            nn.Conv2d(channels[0], out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),
        )

        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Encoder
        e0 = self.firstblock(x)
        e1 = self.down_blocks[0](self.maxpools[0](e0))
        e2 = self.down_blocks[1](self.maxpools[1](e1))
        e3 = self.down_blocks[2](self.maxpools[2](e2))
        e4 = self.maxpools[3](e3)

        # Decoder
        d4 = self.up_blocks[0](e4)

        d3_input = d4 + e3
        d3 = self.up_blocks[1](d3_input)

        d2_input = d3 + e2
        d2 = self.up_blocks[2](d2_input)

        d1_input = d2 + e1
        d1 = self.up_blocks[3](d1_input)

        d1 += e0
        out = self.lastConv(d1)

        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, g=1, act=True):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.03)
        self.act  = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LiteDetBlock(nn.Module):
    """Depthwise-separable + residual, tunad för detection/CPU."""
    def __init__(self, ch, expansion=2.0, use_se=False):
        super().__init__()
        hidden = int(ch * expansion)
        self.pw1 = ConvBNAct(ch, hidden, k=1, s=1, g=1)
        self.dw  = ConvBNAct(hidden, hidden, k=3, s=1, g=hidden)  # depthwise
        self.pw2 = ConvBNAct(hidden, ch, k=1, s=1, g=1, act=False)

        self.use_se = use_se
        if use_se:
            squeeze = max(ch // 8, 4)
            self.se_fc1 = nn.Conv2d(ch, squeeze, 1)
            self.se_fc2 = nn.Conv2d(squeeze, ch, 1)

        self.act = nn.SiLU(inplace=True)

    def _se(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = self.se_fc2(F.silu(self.se_fc1(s)))
        return torch.sigmoid(s) * x

    def forward(self, x):
        identity = x
        out = self.pw2(self.dw(self.pw1(x)))
        if self.use_se:
            out = self._se(out)
        out = out + identity
        return self.act(out)


class LiteDetStage(nn.Module):
    def __init__(self, in_ch, out_ch, stride, n_blocks, expansion, use_se_last=False):
        super().__init__()
        # ned-sampling + kanalökning
        self.down = ConvBNAct(in_ch, out_ch, k=3, s=stride, g=1)
        blocks = []
        for i in range(n_blocks):
            blocks.append(
                LiteDetBlock(
                    out_ch,
                    expansion=expansion,
                    use_se=(use_se_last and i == n_blocks - 1),
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.down(x)
        x = self.blocks(x)
        return x


class LiteDetBackbone(nn.Module):
    """
    Custom CPU-friendly backbone för detection.
    Returnerar feature maps vid strides 4, 8, 16, 32 (C2..C5).
    """
    def __init__(
        self,
        in_chans=3,
        base_channels=(24, 48, 96, 192),
        layers=(1, 2, 3, 2),
        expansions=(2.0, 2.0, 2.0, 2.0),
        use_se_last=(False, True, True, True),
    ):
        super().__init__()
        assert len(base_channels) == len(layers) == len(expansions) == len(use_se_last) == 4

        c2, c3, c4, c5 = base_channels

        # stem → stride 2 (1/2)
        self.stem = ConvBNAct(in_chans, c2 // 2, k=3, s=2)

        # stage2 → stride 2 (1/4)  → C2
        self.stage2 = LiteDetStage(
            c2 // 2, c2,
            stride=2,
            n_blocks=layers[0],
            expansion=expansions[0],
            use_se_last=use_se_last[0],
        )

        # stage3 → stride 2 (1/8)  → C3
        self.stage3 = LiteDetStage(
            c2, c3,
            stride=2,
            n_blocks=layers[1],
            expansion=expansions[1],
            use_se_last=use_se_last[1],
        )

        # stage4 → stride 2 (1/16) → C4
        self.stage4 = LiteDetStage(
            c3, c4,
            stride=2,
            n_blocks=layers[2],
            expansion=expansions[2],
            use_se_last=use_se_last[2],
        )

        # stage5 → stride 2 (1/32) → C5
        self.stage5 = LiteDetStage(
            c4, c5,
            stride=2,
            n_blocks=layers[3],
            expansion=expansions[3],
            use_se_last=use_se_last[3],
        )

        # praktiskt för FPN
        self.out_channels = (c2, c3, c4, c5)

    def forward(self, x):
        x = self.stem(x)    # 1/2
        c2 = self.stage2(x) # 1/4
        c3 = self.stage3(c2) # 1/8
        c4 = self.stage4(c3) # 1/16
        c5 = self.stage5(c4) # 1/32
        return c2, c3, c4, c5


def build_litedet_backbone(variant: str = "edge_n", in_chans: int = 3):
    """
    Factory för ditt detektions-tunade backbone.
    Varianter:
      - edge_n  ~0.5M params (ungefär)
      - edge_s  ~1M  params
      - edge_m  ~2M  params
      - edge_l  ~4M  params
    """
    variant = variant.lower()

    if variant == "edge_n":
        cfg = dict(
            base_channels=(24, 48, 96, 192),
            layers=(1, 2, 3, 2),
            expansions=(1.5, 2.0, 2.0, 2.0),
            use_se_last=(False, True, True, True),
        )
    elif variant == "edge_s":
        cfg = dict(
            base_channels=(32, 64, 128, 224),
            layers=(1, 3, 4, 3),
            expansions=(2.0, 2.0, 2.0, 2.0),
            use_se_last=(False, True, True, True),
        )
    elif variant == "edge_m":
        cfg = dict(
            base_channels=(40, 80, 160, 256),
            layers=(2, 3, 5, 3),
            expansions=(2.0, 2.0, 2.5, 2.5),
            use_se_last=(False, True, True, True),
        )
    elif variant == "edge_l":
        cfg = dict(
            base_channels=(48, 96, 192, 320),
            layers=(2, 4, 6, 4),
            expansions=(2.0, 2.5, 2.5, 2.5),
            use_se_last=(True, True, True, True),
        )
    else:
        raise ValueError(f"Unknown LiteDet backbone variant: {variant}")

    backbone = LiteDetBackbone(in_chans=in_chans, **cfg)
    return backbone, backbone.out_channels

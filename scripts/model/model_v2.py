import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# -------------------------------------------------
# Utils
# -------------------------------------------------

def init_detect_bias(head_moduledict, num_classes, p_obj=0.01):
    """Sätter rimliga start-bias för obj/cls (decoupled heads)."""
    obj_bias = -math.log((1 - p_obj) / p_obj)      # ~ -4.595 för p=0.01
    cls_bias = (-math.log(num_classes)) if num_classes > 1 else 0.0
    with torch.no_grad():
        head_moduledict["out"]["obj"].bias.fill_(obj_bias)
        head_moduledict["out"]["cls"].bias.fill_(cls_bias)
        head_moduledict["out"]["box"].bias.zero_()


def conv_block(c_in, c_out, n=1):
    """Standard conv block (Conv+BN+SiLU) * n."""
    layers = []
    for i in range(n):
        layers.append(nn.Conv2d(c_in if i == 0 else c_out, c_out, 3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(c_out))
        layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)


class DWConvBlock(nn.Module):
    """Depthwise separable conv block med valbar activation, repeterad n ggr."""
    def __init__(self, c_in, c_out, n=1, act="relu"):
        super().__init__()
        if act == "relu":
            act_layer = nn.ReLU
        elif act == "silu":
            act_layer = nn.SiLU
        else:
            raise ValueError(f"Unknown act={act}")

        layers = []
        for i in range(n):
            cin = c_in if i == 0 else c_out
            layers.extend([
                nn.Conv2d(cin, cin, kernel_size=3, padding=1, groups=cin, bias=False),
                nn.Conv2d(cin, c_out, kernel_size=1, bias=False),
                nn.BatchNorm2d(c_out),
                act_layer(inplace=True),
            ])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


def make_head(A, head_depth, C, fpn_channels, head_dw=True, head_act="relu"):
    """
    Head som matchar ditt nuvarande outputformat:
    box: A*4, obj: A*1, cls: A*C
    trunk: (depthwise eller vanlig) blocks
    """
    if head_depth < 1:
        head_depth = 1

    trunk_layers = []
    for _ in range(head_depth):
        if head_dw:
            trunk_layers.append(DWConvBlock(fpn_channels, fpn_channels, n=1, act=head_act))
        else:
            # snabb fallback om du vill: vanlig conv trunk
            trunk_layers.append(nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.SiLU(inplace=True) if head_act == "silu" else nn.ReLU(inplace=True)
            ))
    trunk = nn.Sequential(*trunk_layers)

    out_layers = nn.ModuleDict({
        "box": nn.Conv2d(fpn_channels, A * 4, 1),
        "obj": nn.Conv2d(fpn_channels, A * 1, 1),
        "cls": nn.Conv2d(fpn_channels, A * C, 1),
    })
    return nn.ModuleDict({"trunk": trunk, "out": out_layers})


def _flatten_level_outputs(outs, export_concat: bool):
    if not export_concat:
        return outs
    flat = []
    for p in outs:  # p: [B, A, S, S, E]
        B, A, S, _, E = p.shape
        flat.append(p.view(B, -1, E))
    return torch.cat(flat, dim=1)  # [B, N_total, E]


def _pick_out_indices(feature_info, take: int = 3):
    n = len(feature_info)
    out_idx = list(range(n - take, n))
    reductions = [feature_info[i]["reduction"] for i in out_idx]
    chs = [feature_info[i]["num_chs"] for i in out_idx]
    return out_idx, reductions, chs


# -------------------------------------------------
# Neck building blocks for variants
# -------------------------------------------------

class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=1, s=1, p=None, groups=1, act="silu"):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unknown act={act}")

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Downsample(nn.Module):
    """3x3 stride-2 downsample, valbart DW-separable."""
    def __init__(self, c, dw=False, act="silu"):
        super().__init__()
        if not dw:
            self.m = ConvBNAct(c, c, k=3, s=2, act=act)
        else:
            self.m = nn.Sequential(
                ConvBNAct(c, c, k=3, s=2, groups=c, act=act),
                ConvBNAct(c, c, k=1, s=1, act=act),
            )

    def forward(self, x):
        return self.m(x)


class C2fLite(nn.Module):
    """
    Förenklad C2f: proj -> split -> kedja -> concat -> proj
    Tar in (2*C-ish) och returnerar C.
    """
    def __init__(self, c_in, c_out, n=1, dw=False, act="silu"):
        super().__init__()
        hidden = max(8, c_out // 2)
        self.cv1 = ConvBNAct(c_in, 2 * hidden, k=1, act=act)

        blocks = []
        for _ in range(n):
            if not dw:
                blocks.append(ConvBNAct(hidden, hidden, k=3, act=act))
            else:
                blocks.append(nn.Sequential(
                    ConvBNAct(hidden, hidden, k=3, groups=hidden, act=act),
                    ConvBNAct(hidden, hidden, k=1, act=act),
                ))
        self.m = nn.Sequential(*blocks)
        self.cv2 = ConvBNAct((2 + n) * hidden, c_out, k=1, act=act)

    def forward(self, x):
        y = self.cv1(x)
        y1, y2 = y.chunk(2, dim=1)
        outs = [y1, y2]
        for blk in self.m:
            y2 = blk(y2)
            outs.append(y2)
        return self.cv2(torch.cat(outs, dim=1))


class WeightedAdd(nn.Module):
    """BiFPN-style viktad summa (ReLU-normaliserad)."""
    def __init__(self, n_inputs: int, eps: float = 1e-4):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))
        self.eps = eps

    def forward(self, *xs):
        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)
        out = 0.0
        for i, x in enumerate(xs):
            out = out + w[i] * x
        return out


# -------------------------------------------------
# Model (GPU/SiLU neck by default, head output is same format)
# -------------------------------------------------

class YOLOLiteMS(nn.Module):
    def __init__(
        self,
        backbone="resnet18",
        num_classes=3,
        fpn_channels=128,
        num_anchors_per_level=(3, 3, 3, 3),
        pretrained=True,
        depth_multiple: float = 1.0,
        width_multiple: float = 1.0,
        head_depth: int = 1,
        use_p6: bool = True,
        use_p2: bool = False,
        neck_variant: str = "fpn_add",  # "fpn_add" (nuvarande), "pan_concat", "bifpn_lite"
    ):
        super().__init__()

        self.use_p6 = use_p6
        self.use_p2 = use_p2
        self.neck_variant = neck_variant

        # Probe backbone once to learn channels/reductions
        tmp = timm.create_model(backbone, features_only=True, pretrained=pretrained)
        take = 4 if use_p2 else 3
        out_idx, reductions, chs = _pick_out_indices(tmp.feature_info, take=take)

        self.backbone = timm.create_model(
            backbone, features_only=True, pretrained=pretrained, out_indices=out_idx
        )
        self.reductions = reductions

        # Width/Depth scaling
        fpn_channels = int(fpn_channels * width_multiple)
        d = max(1, round(2 * depth_multiple))

        # Unpack backbone feature channels
        if self.use_p2:
            c2, c3, c4, c5 = chs
        else:
            c3, c4, c5 = chs

        # Laterals
        if self.use_p2:
            self.lateral2 = nn.Conv2d(c2, fpn_channels, 1)
        self.lateral3 = nn.Conv2d(c3, fpn_channels, 1)
        self.lateral4 = nn.Conv2d(c4, fpn_channels, 1)
        self.lateral5 = nn.Conv2d(c5, fpn_channels, 1)

        # --- Neck modules ---
        if self.neck_variant == "fpn_add":
            # Din nuvarande FPN-add
            if self.use_p2:
                self.smooth2 = conv_block(fpn_channels, fpn_channels, n=d)
            self.smooth3 = conv_block(fpn_channels, fpn_channels, n=d)
            self.smooth4 = conv_block(fpn_channels, fpn_channels, n=d)
            self.smooth5 = conv_block(fpn_channels, fpn_channels, n=d)

        elif self.neck_variant == "pan_concat":
            # YOLOv8-ish PAN-FPN med concat + C2fLite
            act = "silu"
            dw = False  # kan sättas True om du vill mer CPU-likt på GPU också
            n_fuse = d

            # vi använder en "pre-smooth" på p5 för stabilitet
            self.smooth5 = conv_block(fpn_channels, fpn_channels, n=1)

            self.fuse4_td = C2fLite(2 * fpn_channels, fpn_channels, n=n_fuse, dw=dw, act=act)
            self.fuse3_td = C2fLite(2 * fpn_channels, fpn_channels, n=n_fuse, dw=dw, act=act)

            if self.use_p2:
                self.fuse2_td = C2fLite(2 * fpn_channels, fpn_channels, n=n_fuse, dw=dw, act=act)
                self.down2 = Downsample(fpn_channels, dw=dw, act=act)
                self.fuse3_bu = C2fLite(2 * fpn_channels, fpn_channels, n=n_fuse, dw=dw, act=act)

            self.down3 = Downsample(fpn_channels, dw=dw, act=act)
            self.down4 = Downsample(fpn_channels, dw=dw, act=act)
            self.fuse4_bu = C2fLite(2 * fpn_channels, fpn_channels, n=n_fuse, dw=dw, act=act)
            self.fuse5_bu = C2fLite(2 * fpn_channels, fpn_channels, n=n_fuse, dw=dw, act=act)

        elif self.neck_variant == "bifpn_lite":
            # BiFPN-lite med WeightedAdd + smooth blocks
            act = "silu"
            dw = False

            self.smooth5 = conv_block(fpn_channels, fpn_channels, n=1)

            # top-down weights
            self.w4_td = WeightedAdd(2)
            self.w3_td = WeightedAdd(2)
            if self.use_p2:
                self.w2_td = WeightedAdd(2)

            # bottom-up weights
            if self.use_p2:
                self.w3_bu = WeightedAdd(2)
                self.down2 = Downsample(fpn_channels, dw=dw, act=act)
                self.b3_bu = conv_block(fpn_channels, fpn_channels, n=d)

            self.down3 = Downsample(fpn_channels, dw=dw, act=act)
            self.down4 = Downsample(fpn_channels, dw=dw, act=act)
            self.w4_bu = WeightedAdd(2)
            self.w5_bu = WeightedAdd(2)

            # smooth after fusion
            self.b4_td = conv_block(fpn_channels, fpn_channels, n=d)
            self.b3_td = conv_block(fpn_channels, fpn_channels, n=d)
            if self.use_p2:
                self.b2_td = conv_block(fpn_channels, fpn_channels, n=d)
            self.b4_bu = conv_block(fpn_channels, fpn_channels, n=d)
            self.b5_bu = conv_block(fpn_channels, fpn_channels, n=d)

        else:
            raise ValueError(f"Unknown neck_variant={self.neck_variant}")

        # P6 path (behåll din stil – används oavsett variant, men bara i forward om use_p6)
        self.p6_down = nn.Conv2d(fpn_channels, fpn_channels, 3, 2, 1, bias=False)
        self.p6_bn = nn.BatchNorm2d(fpn_channels)
        self.p6_act = nn.SiLU(inplace=True)
        self.smooth6 = conv_block(fpn_channels, fpn_channels, n=d)

        # Anchors per level mapping
        C = int(num_classes)
        level_names = (["p2"] if self.use_p2 else []) + ["p3", "p4", "p5"] + (["p6"] if self.use_p6 else [])
        if len(num_anchors_per_level) >= 3:
            A3, A4, A5 = map(int, num_anchors_per_level[:3])
            A2 = A3
            A6 = A5
        else:
            A2 = A3 = A4 = A5 = A6 = int(num_anchors_per_level[0]) if len(num_anchors_per_level) else 1

        anchors_map = {"p2": A2, "p3": A3, "p4": A4, "p5": A5, "p6": A6}
        self.num_anchors_per_level = tuple(anchors_map[n] for n in level_names)
        self.num_classes = C

        # Export switches
        self.export_concat = False
        self.export_decode = False

        # Heads (behåll format)
        head_dw = True
        head_act = "relu"  # kan sättas till "silu" om du vill, men jag lämnar nära din original
        if self.use_p2:
            self.head2 = make_head(anchors_map["p2"], head_depth, C, fpn_channels, head_dw=head_dw, head_act=head_act)
            init_detect_bias(self.head2, C)
        self.head3 = make_head(anchors_map["p3"], head_depth, C, fpn_channels, head_dw=head_dw, head_act=head_act)
        self.head4 = make_head(anchors_map["p4"], head_depth, C, fpn_channels, head_dw=head_dw, head_act=head_act)
        self.head5 = make_head(anchors_map["p5"], head_depth, C, fpn_channels, head_dw=head_dw, head_act=head_act)
        init_detect_bias(self.head3, C)
        init_detect_bias(self.head4, C)
        init_detect_bias(self.head5, C)
        if self.use_p6:
            self.head6 = make_head(anchors_map["p6"], head_depth, C, fpn_channels, head_dw=head_dw, head_act=head_act)
            init_detect_bias(self.head6, C)

        # Self-describing strides
        base = list(self.reductions)
        self.fpn_strides = base + ([base[-1] * 2] if self.use_p6 else [])

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.shape[-2:], mode="nearest") + y

    def _forward_head(self, p, head_dict, A):
        p = head_dict["trunk"](p)
        box = head_dict["out"]["box"](p)
        obj = head_dict["out"]["obj"](p)
        cls = head_dict["out"]["cls"](p)
        B, _, S, _ = box.shape
        box = box.view(B, A, 4, S, S)
        obj = obj.view(B, A, 1, S, S)
        cls = cls.view(B, A, self.num_classes, S, S)
        out = torch.cat([box, obj, cls], dim=2)
        return out.permute(0, 1, 3, 4, 2).contiguous()

    def forward(self, x):
        feats = self.backbone(x)
        if self.use_p2:
            c2, c3, c4, c5 = feats
        else:
            c3, c4, c5 = feats

        # --- Build neck outputs p2/p3/p4/p5 ---
        if self.neck_variant == "fpn_add":
            p5_out = self.smooth5(self.lateral5(c5))
            p4_out = self.smooth4(self._upsample_add(p5_out, self.lateral4(c4)))
            p3_out = self.smooth3(self._upsample_add(p4_out, self.lateral3(c3)))
            p2_out = None
            if self.use_p2:
                p2_out = self.smooth2(self._upsample_add(p3_out, self.lateral2(c2)))

        elif self.neck_variant == "pan_concat":
            p5_in = self.smooth5(self.lateral5(c5))
            p4_in = self.lateral4(c4)
            p3_in = self.lateral3(c3)

            p4_td = self.fuse4_td(torch.cat([F.interpolate(p5_in, size=p4_in.shape[-2:], mode="nearest"), p4_in], dim=1))
            p3_td = self.fuse3_td(torch.cat([F.interpolate(p4_td, size=p3_in.shape[-2:], mode="nearest"), p3_in], dim=1))

            p2_out = None
            if self.use_p2:
                p2_in = self.lateral2(c2)
                p2_td = self.fuse2_td(torch.cat([F.interpolate(p3_td, size=p2_in.shape[-2:], mode="nearest"), p2_in], dim=1))
                p3_out = self.fuse3_bu(torch.cat([self.down2(p2_td), p3_td], dim=1))
                p2_out = p2_td
            else:
                p3_out = p3_td

            p4_out = self.fuse4_bu(torch.cat([self.down3(p3_out), p4_td], dim=1))
            p5_out = self.fuse5_bu(torch.cat([self.down4(p4_out), p5_in], dim=1))

        elif self.neck_variant == "bifpn_lite":
            p5_in = self.smooth5(self.lateral5(c5))
            p4_in = self.lateral4(c4)
            p3_in = self.lateral3(c3)

            p4_td = self.b4_td(self.w4_td(F.interpolate(p5_in, size=p4_in.shape[-2:], mode="nearest"), p4_in))
            p3_td = self.b3_td(self.w3_td(F.interpolate(p4_td, size=p3_in.shape[-2:], mode="nearest"), p3_in))

            p2_out = None
            if self.use_p2:
                p2_in = self.lateral2(c2)
                p2_td = self.b2_td(self.w2_td(F.interpolate(p3_td, size=p2_in.shape[-2:], mode="nearest"), p2_in))
                p3_out = self.b3_bu(self.w3_bu(self.down2(p2_td), p3_td))
                p2_out = p2_td
            else:
                p3_out = p3_td

            p4_out = self.b4_bu(self.w4_bu(self.down3(p3_out), p4_td))
            p5_out = self.b5_bu(self.w5_bu(self.down4(p4_out), p5_in))

        else:
            raise RuntimeError("neck_variant should have been validated in __init__")

        # --- Heads ---
        outs = []
        if self.use_p2:
            outs.append(self._forward_head(p2_out, self.head2, self.num_anchors_per_level[0]))

        idx = 1 if self.use_p2 else 0
        outs.append(self._forward_head(p3_out, self.head3, self.num_anchors_per_level[idx + 0]))
        outs.append(self._forward_head(p4_out, self.head4, self.num_anchors_per_level[idx + 1]))
        outs.append(self._forward_head(p5_out, self.head5, self.num_anchors_per_level[idx + 2]))

        if self.use_p6:
            p6 = self.smooth6(self.p6_act(self.p6_bn(self.p6_down(p5_out))))
            outs.append(self._forward_head(p6, self.head6, self.num_anchors_per_level[idx + 3]))

        return _flatten_level_outputs(outs, self.export_concat)

    def get_strides(self):
        return list(self.fpn_strides)

    def get_num_anchors_per_level(self):
        return tuple(self.num_anchors_per_level)

    def print_strides(self, img_size=640):
        with torch.no_grad():
            d = next(self.parameters()).device
            x = torch.zeros(1, 3, img_size, img_size, device=d)
            feats = self.backbone(x)
            if self.use_p2:
                c2, c3, c4, c5 = feats
                Ss = [c2.shape[-1], c3.shape[-1], c4.shape[-1], c5.shape[-1]]
            else:
                c3, c4, c5 = feats
                Ss = [c3.shape[-1], c4.shape[-1], c5.shape[-1]]
            if self.use_p6:
                Ss.append(Ss[-1] // 2)
            strides = [img_size // S for S in Ss]
            print(f"[YOLOLiteMS] grids={Ss} → strides={strides}")


# -------------------------------------------------
# Model CPU (DW+ReLU neck/head, same output format)
# -------------------------------------------------

class YOLOLiteMS_CPU(nn.Module):
    def __init__(
        self,
        backbone="mobilenetv3_small_100",
        num_classes=3,
        fpn_channels=96,
        num_anchors_per_level=(3, 3, 3, 3),
        pretrained=True,
        depth_multiple=0.75,
        width_multiple=0.75,
        head_depth=1,
        use_p6: bool = True,
        use_p2: bool = False,
        neck_variant: str = "fpn_add",  # "fpn_add", "pan_concat", "bifpn_lite"
    ):
        super().__init__()

        self.use_p6 = use_p6
        self.use_p2 = use_p2
        self.neck_variant = neck_variant

        tmp = timm.create_model(backbone, features_only=True, pretrained=pretrained)
        take = 4 if use_p2 else 3
        out_idx, reductions, chs = _pick_out_indices(tmp.feature_info, take=take)

        self.backbone = timm.create_model(
            backbone, features_only=True, pretrained=pretrained, out_indices=out_idx
        )
        self.reductions = reductions

        fpn_channels = int(fpn_channels * width_multiple)
        d = max(1, round(2 * depth_multiple))

        if self.use_p2:
            c2, c3, c4, c5 = chs
        else:
            c3, c4, c5 = chs

        # Laterals
        if self.use_p2:
            self.lateral2 = nn.Conv2d(c2, fpn_channels, 1)
        self.lateral3 = nn.Conv2d(c3, fpn_channels, 1)
        self.lateral4 = nn.Conv2d(c4, fpn_channels, 1)
        self.lateral5 = nn.Conv2d(c5, fpn_channels, 1)

        # Neck
        if self.neck_variant == "fpn_add":
            if self.use_p2:
                self.smooth2 = DWConvBlock(fpn_channels, fpn_channels, n=d, act="relu")
            self.smooth3 = DWConvBlock(fpn_channels, fpn_channels, n=d, act="relu")
            self.smooth4 = DWConvBlock(fpn_channels, fpn_channels, n=d, act="relu")
            self.smooth5 = DWConvBlock(fpn_channels, fpn_channels, n=d, act="relu")

        elif self.neck_variant == "pan_concat":
            act = "relu"
            dw = True
            n_fuse = d

            self.smooth5 = DWConvBlock(fpn_channels, fpn_channels, n=1, act=act)

            self.fuse4_td = C2fLite(2 * fpn_channels, fpn_channels, n=n_fuse, dw=dw, act=act)
            self.fuse3_td = C2fLite(2 * fpn_channels, fpn_channels, n=n_fuse, dw=dw, act=act)

            if self.use_p2:
                self.fuse2_td = C2fLite(2 * fpn_channels, fpn_channels, n=n_fuse, dw=dw, act=act)
                self.down2 = Downsample(fpn_channels, dw=dw, act=act)
                self.fuse3_bu = C2fLite(2 * fpn_channels, fpn_channels, n=n_fuse, dw=dw, act=act)

            self.down3 = Downsample(fpn_channels, dw=dw, act=act)
            self.down4 = Downsample(fpn_channels, dw=dw, act=act)
            self.fuse4_bu = C2fLite(2 * fpn_channels, fpn_channels, n=n_fuse, dw=dw, act=act)
            self.fuse5_bu = C2fLite(2 * fpn_channels, fpn_channels, n=n_fuse, dw=dw, act=act)

        elif self.neck_variant == "bifpn_lite":
            act = "relu"
            dw = True

            self.smooth5 = DWConvBlock(fpn_channels, fpn_channels, n=1, act=act)

            self.w4_td = WeightedAdd(2)
            self.w3_td = WeightedAdd(2)
            if self.use_p2:
                self.w2_td = WeightedAdd(2)

            if self.use_p2:
                self.w3_bu = WeightedAdd(2)
                self.down2 = Downsample(fpn_channels, dw=dw, act=act)
                self.b3_bu = DWConvBlock(fpn_channels, fpn_channels, n=d, act=act)

            self.down3 = Downsample(fpn_channels, dw=dw, act=act)
            self.down4 = Downsample(fpn_channels, dw=dw, act=act)
            self.w4_bu = WeightedAdd(2)
            self.w5_bu = WeightedAdd(2)

            self.b4_td = DWConvBlock(fpn_channels, fpn_channels, n=d, act=act)
            self.b3_td = DWConvBlock(fpn_channels, fpn_channels, n=d, act=act)
            if self.use_p2:
                self.b2_td = DWConvBlock(fpn_channels, fpn_channels, n=d, act=act)
            self.b4_bu = DWConvBlock(fpn_channels, fpn_channels, n=d, act=act)
            self.b5_bu = DWConvBlock(fpn_channels, fpn_channels, n=d, act=act)

        else:
            raise ValueError(f"Unknown neck_variant={self.neck_variant}")

        # P6 path
        self.p6_down = nn.Conv2d(fpn_channels, fpn_channels, 3, 2, 1, bias=False)
        self.p6_bn = nn.BatchNorm2d(fpn_channels)
        self.p6_act = nn.ReLU(inplace=True)
        self.smooth6 = DWConvBlock(fpn_channels, fpn_channels, n=d, act="relu")

        # Anchors per level mapping
        C = int(num_classes)
        level_names = (["p2"] if self.use_p2 else []) + ["p3", "p4", "p5"] + (["p6"] if self.use_p6 else [])
        if len(num_anchors_per_level) >= 3:
            A3, A4, A5 = map(int, num_anchors_per_level[:3])
            A2 = A3
            A6 = A5
        else:
            A2 = A3 = A4 = A5 = A6 = int(num_anchors_per_level[0]) if len(num_anchors_per_level) else 1

        anchors_map = {"p2": A2, "p3": A3, "p4": A4, "p5": A5, "p6": A6}
        self.num_anchors_per_level = tuple(anchors_map[n] for n in level_names)
        self.num_classes = C

        self.export_concat = False
        self.export_decode = False

        # Heads
        head_dw = True
        head_act = "relu"
        if self.use_p2:
            self.head2 = make_head(anchors_map["p2"], head_depth, C, fpn_channels, head_dw=head_dw, head_act=head_act)
            init_detect_bias(self.head2, C)
        self.head3 = make_head(anchors_map["p3"], head_depth, C, fpn_channels, head_dw=head_dw, head_act=head_act)
        self.head4 = make_head(anchors_map["p4"], head_depth, C, fpn_channels, head_dw=head_dw, head_act=head_act)
        self.head5 = make_head(anchors_map["p5"], head_depth, C, fpn_channels, head_dw=head_dw, head_act=head_act)
        init_detect_bias(self.head3, C)
        init_detect_bias(self.head4, C)
        init_detect_bias(self.head5, C)
        if self.use_p6:
            self.head6 = make_head(anchors_map["p6"], head_depth, C, fpn_channels, head_dw=head_dw, head_act=head_act)
            init_detect_bias(self.head6, C)

        base = list(self.reductions)
        self.fpn_strides = base + ([base[-1] * 2] if self.use_p6 else [])

    def _forward_head(self, p, head_dict, A):
        p = head_dict["trunk"](p)
        box = head_dict["out"]["box"](p)
        obj = head_dict["out"]["obj"](p)
        cls = head_dict["out"]["cls"](p)
        B, _, S, _ = box.shape
        box = box.view(B, A, 4, S, S)
        obj = obj.view(B, A, 1, S, S)
        cls = cls.view(B, A, self.num_classes, S, S)
        out = torch.cat([box, obj, cls], dim=2)
        return out.permute(0, 1, 3, 4, 2).contiguous()

    def forward(self, x):
        feats = self.backbone(x)
        if self.use_p2:
            c2, c3, c4, c5 = feats
        else:
            c3, c4, c5 = feats

        if self.neck_variant == "fpn_add":
            p5_out = self.smooth5(self.lateral5(c5))
            p4_out = self.smooth4(F.interpolate(p5_out, size=self.lateral4(c4).shape[-2:], mode="nearest") + self.lateral4(c4))
            p3_out = self.smooth3(F.interpolate(p4_out, size=self.lateral3(c3).shape[-2:], mode="nearest") + self.lateral3(c3))
            p2_out = None
            if self.use_p2:
                p2_out = self.smooth2(F.interpolate(p3_out, size=self.lateral2(c2).shape[-2:], mode="nearest") + self.lateral2(c2))

        elif self.neck_variant == "pan_concat":
            p5_in = self.smooth5(self.lateral5(c5))
            p4_in = self.lateral4(c4)
            p3_in = self.lateral3(c3)

            p4_td = self.fuse4_td(torch.cat([F.interpolate(p5_in, size=p4_in.shape[-2:], mode="nearest"), p4_in], dim=1))
            p3_td = self.fuse3_td(torch.cat([F.interpolate(p4_td, size=p3_in.shape[-2:], mode="nearest"), p3_in], dim=1))

            p2_out = None
            if self.use_p2:
                p2_in = self.lateral2(c2)
                p2_td = self.fuse2_td(torch.cat([F.interpolate(p3_td, size=p2_in.shape[-2:], mode="nearest"), p2_in], dim=1))
                p3_out = self.fuse3_bu(torch.cat([self.down2(p2_td), p3_td], dim=1))
                p2_out = p2_td
            else:
                p3_out = p3_td

            p4_out = self.fuse4_bu(torch.cat([self.down3(p3_out), p4_td], dim=1))
            p5_out = self.fuse5_bu(torch.cat([self.down4(p4_out), p5_in], dim=1))

        elif self.neck_variant == "bifpn_lite":
            p5_in = self.smooth5(self.lateral5(c5))
            p4_in = self.lateral4(c4)
            p3_in = self.lateral3(c3)

            p4_td = self.b4_td(self.w4_td(F.interpolate(p5_in, size=p4_in.shape[-2:], mode="nearest"), p4_in))
            p3_td = self.b3_td(self.w3_td(F.interpolate(p4_td, size=p3_in.shape[-2:], mode="nearest"), p3_in))

            p2_out = None
            if self.use_p2:
                p2_in = self.lateral2(c2)
                p2_td = self.b2_td(self.w2_td(F.interpolate(p3_td, size=p2_in.shape[-2:], mode="nearest"), p2_in))
                p3_out = self.b3_bu(self.w3_bu(self.down2(p2_td), p3_td))
                p2_out = p2_td
            else:
                p3_out = p3_td

            p4_out = self.b4_bu(self.w4_bu(self.down3(p3_out), p4_td))
            p5_out = self.b5_bu(self.w5_bu(self.down4(p4_out), p5_in))

        else:
            raise RuntimeError("neck_variant should have been validated in __init__")

        outs = []
        if self.use_p2:
            outs.append(self._forward_head(p2_out, self.head2, self.num_anchors_per_level[0]))

        idx = 1 if self.use_p2 else 0
        outs.append(self._forward_head(p3_out, self.head3, self.num_anchors_per_level[idx + 0]))
        outs.append(self._forward_head(p4_out, self.head4, self.num_anchors_per_level[idx + 1]))
        outs.append(self._forward_head(p5_out, self.head5, self.num_anchors_per_level[idx + 2]))

        if self.use_p6:
            p6 = self.smooth6(self.p6_act(self.p6_bn(self.p6_down(p5_out))))
            outs.append(self._forward_head(p6, self.head6, self.num_anchors_per_level[idx + 3]))

        return _flatten_level_outputs(outs, self.export_concat)

    def get_strides(self):
        return list(self.fpn_strides)

    def get_num_anchors_per_level(self):
        return tuple(self.num_anchors_per_level)

    def print_strides(self, img_size=640):
        with torch.no_grad():
            d = next(self.parameters()).device
            x = torch.zeros(1, 3, img_size, img_size, device=d)
            feats = self.backbone(x)
            if self.use_p2:
                c2, c3, c4, c5 = feats
                Ss = [c2.shape[-1], c3.shape[-1], c4.shape[-1], c5.shape[-1]]
            else:
                c3, c4, c5 = feats
                Ss = [c3.shape[-1], c4.shape[-1], c5.shape[-1]]
            if self.use_p6:
                Ss.append(Ss[-1] // 2)
            strides = [img_size // S for S in Ss]
            print(f"[YOLOLiteMS_CPU] grids={Ss} → strides={strides}")


# -------------------------------------------------
# Convenience classes (v2/v3)
# -------------------------------------------------

class YOLOLiteMS_v2(YOLOLiteMS):
    """PAN-FPN concat + C2fLite."""
    def __init__(self, *args, **kwargs):
        kwargs["neck_variant"] = "pan_concat"
        super().__init__(*args, **kwargs)


class YOLOLiteMS_v3(YOLOLiteMS):
    """BiFPN-lite weighted add."""
    def __init__(self, *args, **kwargs):
        kwargs["neck_variant"] = "bifpn_lite"
        super().__init__(*args, **kwargs)


class YOLOLiteMS_CPU_v2(YOLOLiteMS_CPU):
    """CPU: PAN-FPN concat + C2fLite (DW)."""
    def __init__(self, *args, **kwargs):
        kwargs["neck_variant"] = "pan_concat"
        super().__init__(*args, **kwargs)


class YOLOLiteMS_CPU_v3(YOLOLiteMS_CPU):
    """CPU: BiFPN-lite weighted add."""
    def __init__(self, *args, **kwargs):
        kwargs["neck_variant"] = "bifpn_lite"
        super().__init__(*args, **kwargs)

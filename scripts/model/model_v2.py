import os, sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from scripts.model.backbone import build_litedet_backbone


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# Viktigt: du lägger själv till något i stil med:
# from scripts.model.backbones.litedet import build_litedet_backbone
# eller motsvarande sökväg i ditt repo.


def init_detect_bias(head_moduledict, num_classes, p_obj=0.01):
    """Sätter rimliga start-bias för obj/cls (decoupled heads)."""
    obj_bias = -math.log((1 - p_obj) / p_obj)      # ~ -4.595 för p=0.01
    cls_bias = (-math.log(num_classes)) if num_classes > 1 else 0.0
    with torch.no_grad():
        head_moduledict["out"]["obj"].bias.fill_(obj_bias)
        head_moduledict["out"]["cls"].bias.fill_(cls_bias)
        head_moduledict["out"]["box"].bias.zero_()


def conv_block(c_in, c_out, n=1):
    """Bygger n st conv-baserade block (3x3 conv + BN + SiLU)."""
    layers = []
    for i in range(n):
        layers.append(nn.Conv2d(c_in if i == 0 else c_out, c_out, 3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(c_out))
        layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)


class DWConvBlock(nn.Module):
    """Depthwise separable conv block med ReLU (CPU-vänlig)."""
    def __init__(self, c_in, c_out, n=1):
        super().__init__()
        layers = []
        for i in range(n):
            in_ch = c_in if i == 0 else c_out
            # depthwise 3x3
            layers.extend([
                nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1,
                          groups=in_ch, bias=False),
                nn.Conv2d(in_ch, c_out, kernel_size=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


def make_head(A, head_depth, C, fpn_channels):
    """
    Detection-head:
      - trunk: ett par DWConvBlock
      - out: separata 1x1-konv för box/obj/cls
    Output per nivå: [B, A, S, S, 5+C]
    """
    trunk = nn.Sequential(*[
        DWConvBlock(fpn_channels, fpn_channels) for _ in range(head_depth)
    ])

    out_layers = nn.ModuleDict({
        "box": nn.Conv2d(fpn_channels, A * 4, 1),
        "obj": nn.Conv2d(fpn_channels, A * 1, 1),
        "cls": nn.Conv2d(fpn_channels, A * C, 1),
    })
    return nn.ModuleDict({"trunk": trunk, "out": out_layers})


def _flatten_level_outputs(outs, export_concat: bool):
    """
    outs: lista av [B, A, S, S, E]
    Om export_concat=True → [B, N_tot, E], annars lista returneras.
    """
    if not export_concat:
        return outs
    flat = []
    for p in outs:
        B, A, S, _, E = p.shape
        flat.append(p.view(B, -1, E))
    return torch.cat(flat, dim=1)


def _pick_out_indices(feature_info, take: int = 3):
    """
    Plocka sista `take` feature maps ur timm's feature_info.
    Returnerar:
      - out_idx: index
      - reductions: stride/reductions (t.ex. [8,16,32])
      - chs: kanaler per nivå
    """
    n = len(feature_info)
    out_idx = list(range(n - take, n))
    reductions = [feature_info[i]["reduction"] for i in out_idx]
    chs = [feature_info[i]["num_chs"] for i in out_idx]
    return out_idx, reductions, chs


class YOLOLiteMS(nn.Module):
    """
    GPU-optimerad variant (vanlig conv-block FPN).
    Stödjer:
      - valfri timm-backbone via features_only
      - special: backbone-namn som börjar med "litedet_" → build_litedet_backbone()
    """
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
        use_p2: bool = False,   # P2-head (stride 4)
    ):
        super().__init__()

        # --- Backbone-setup (timm eller litedet_*) ---
        if isinstance(backbone, str) and backbone.startswith("litedet_"):
            # Du ser till att build_litedet_backbone finns via import högst upp
            variant = backbone.replace("litedet_", "")
            self.backbone, chs, reductions = build_litedet_backbone(variant, in_chans=3)
            self.reductions = reductions
        else:
            tmp = timm.create_model(backbone, features_only=True, pretrained=pretrained)
            take = 4 if use_p2 else 3  # C2..C5 eller C3..C5
            out_idx, reductions, chs = _pick_out_indices(tmp.feature_info, take=take)

            self.backbone = timm.create_model(
                backbone, features_only=True, pretrained=pretrained, out_indices=out_idx
            )
            self.reductions = reductions  # t.ex. [4,8,16,32] eller [8,16,32]

        self.use_p6 = use_p6
        self.use_p2 = use_p2

        # Width/Depth scaling för FPN/head
        fpn_channels = int(fpn_channels * width_multiple)
        d = max(1, round(2 * depth_multiple))

        # Backbone-kanaler → FPN-ingångar
        if self.use_p2:
            c2, c3, c4, c5 = chs
        else:
            c3, c4, c5 = chs

        # FPN laterals
        if self.use_p2:
            self.lateral2 = nn.Conv2d(c2, fpn_channels, 1)
        self.lateral3 = nn.Conv2d(c3, fpn_channels, 1)
        self.lateral4 = nn.Conv2d(c4, fpn_channels, 1)
        self.lateral5 = nn.Conv2d(c5, fpn_channels, 1)

        # FPN smooth blocks (vanliga conv-block, GPU-vänliga)
        if self.use_p2:
            self.smooth2 = conv_block(fpn_channels, fpn_channels, n=d)
        self.smooth3 = conv_block(fpn_channels, fpn_channels, n=d)
        self.smooth4 = conv_block(fpn_channels, fpn_channels, n=d)
        self.smooth5 = conv_block(fpn_channels, fpn_channels, n=d)

        # P6-path (extra lågupplöst nivå)
        self.p6_down = nn.Conv2d(fpn_channels, fpn_channels, 3, 2, 1, bias=False)
        self.p6_bn   = nn.BatchNorm2d(fpn_channels)
        self.p6_act  = nn.SiLU(inplace=True)
        self.smooth6 = conv_block(fpn_channels, fpn_channels, n=d)

        # Anchors per nivå
        C = num_classes
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

        # Heads
        if self.use_p2:
            self.head2 = make_head(anchors_map["p2"], head_depth, C, fpn_channels)
            init_detect_bias(self.head2, C)
        self.head3 = make_head(anchors_map["p3"], head_depth, C, fpn_channels)
        self.head4 = make_head(anchors_map["p4"], head_depth, C, fpn_channels)
        self.head5 = make_head(anchors_map["p5"], head_depth, C, fpn_channels)
        init_detect_bias(self.head3, C)
        init_detect_bias(self.head4, C)
        init_detect_bias(self.head5, C)
        if self.use_p6:
            self.head6 = make_head(anchors_map["p6"], head_depth, C, fpn_channels)
            init_detect_bias(self.head6, C)

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
        feats = self.backbone(x)  # timm: lista; litedet: tuple
        if self.use_p2:
            c2, c3, c4, c5 = feats
        else:
            c3, c4, c5 = feats

        p5 = self.smooth5(self.lateral5(c5))
        p4 = self.smooth4(self._upsample_add(p5, self.lateral4(c4)))
        p3 = self.smooth3(self._upsample_add(p4, self.lateral3(c3)))

        outs = []
        if self.use_p2:
            p2 = self.smooth2(self._upsample_add(p3, self.lateral2(c2)))
            outs.append(self._forward_head(p2, self.head2, self.num_anchors_per_level[0]))

        idx = 1 if self.use_p2 else 0
        outs.append(self._forward_head(p3, self.head3, self.num_anchors_per_level[idx + 0]))
        outs.append(self._forward_head(p4, self.head4, self.num_anchors_per_level[idx + 1]))
        outs.append(self._forward_head(p5, self.head5, self.num_anchors_per_level[idx + 2]))

        if self.use_p6:
            p6 = self.smooth6(self.p6_act(self.p6_bn(self.p6_down(p5))))
            outs.append(self._forward_head(p6, self.head6, self.num_anchors_per_level[idx + 3]))

        return _flatten_level_outputs(outs, self.export_concat)

    # --- Export/script helpers ---
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


class YOLOLiteMS_CPU(nn.Module):
    """
    CPU-optimerad variant:
      - DWConvBlock i FPN + heads
      - Stödjer timm-backbones + litedet_* backbones
    """
    def __init__(
        self,
        backbone="mobilenetv3_small_100",
        num_classes=3,
        fpn_channels=96,
        num_anchors_per_level=(3, 3, 3, 3),
        pretrained=True,
        depth_multiple: float = 0.75,
        width_multiple: float = 0.75,
        head_depth: int = 1,
        use_p6: bool = True,
        use_p2: bool = False,
    ):
        super().__init__()

        # --- Backbone-setup (timm eller litedet_*) ---
        if isinstance(backbone, str) and backbone.startswith("litedet_"):
            variant = backbone.replace("litedet_", "")
            self.backbone, chs, reductions = build_litedet_backbone(variant, in_chans=3)
            self.reductions = reductions
        else:
            tmp = timm.create_model(backbone, features_only=True, pretrained=pretrained)
            take = 4 if use_p2 else 3
            out_idx, reductions, chs = _pick_out_indices(tmp.feature_info, take=take)

            self.backbone = timm.create_model(
                backbone, features_only=True, pretrained=pretrained, out_indices=out_idx
            )
            self.reductions = reductions

        self.use_p6 = use_p6
        self.use_p2 = use_p2

        fpn_channels = int(fpn_channels * width_multiple)
        d = max(1, round(2 * depth_multiple))

        if self.use_p2:
            c2, c3, c4, c5 = chs
        else:
            c3, c4, c5 = chs

        # DW-friendly FPN
        if self.use_p2:
            self.lateral2 = nn.Conv2d(c2, fpn_channels, 1)
            self.smooth2  = DWConvBlock(fpn_channels, fpn_channels, n=d)
        self.lateral3 = nn.Conv2d(c3, fpn_channels, 1)
        self.lateral4 = nn.Conv2d(c4, fpn_channels, 1)
        self.lateral5 = nn.Conv2d(c5, fpn_channels, 1)
        self.smooth3  = DWConvBlock(fpn_channels, fpn_channels, n=d)
        self.smooth4  = DWConvBlock(fpn_channels, fpn_channels, n=d)
        self.smooth5  = DWConvBlock(fpn_channels, fpn_channels, n=d)

        # P6 DW path
        self.p6_down  = nn.Conv2d(fpn_channels, fpn_channels, 3, 2, 1, bias=False)
        self.p6_bn    = nn.BatchNorm2d(fpn_channels)
        self.p6_act   = nn.ReLU(inplace=True)
        self.smooth6  = DWConvBlock(fpn_channels, fpn_channels, n=d)

        # Anchors per level
        C = num_classes
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

        # Export flags
        self.export_concat = False
        self.export_decode = False

        # Heads
        if self.use_p2:
            self.head2 = make_head(anchors_map["p2"], head_depth, C, fpn_channels)
            init_detect_bias(self.head2, C)
        self.head3 = make_head(anchors_map["p3"], head_depth, C, fpn_channels)
        self.head4 = make_head(anchors_map["p4"], head_depth, C, fpn_channels)
        self.head5 = make_head(anchors_map["p5"], head_depth, C, fpn_channels)
        init_detect_bias(self.head3, C)
        init_detect_bias(self.head4, C)
        init_detect_bias(self.head5, C)
        if self.use_p6:
            self.head6 = make_head(anchors_map["p6"], head_depth, C, fpn_channels)
            init_detect_bias(self.head6, C)

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

        p5 = self.smooth5(self.lateral5(c5))
        p4 = self.smooth4(self._upsample_add(p5, self.lateral4(c4)))
        p3 = self.smooth3(self._upsample_add(p4, self.lateral3(c3)))

        outs = []
        if self.use_p2:
            p2 = self.smooth2(self._upsample_add(p3, self.lateral2(c2)))
            outs.append(self._forward_head(p2, self.head2, self.num_anchors_per_level[0]))

        idx = 1 if self.use_p2 else 0
        outs.append(self._forward_head(p3, self.head3, self.num_anchors_per_level[idx + 0]))
        outs.append(self._forward_head(p4, self.head4, self.num_anchors_per_level[idx + 1]))
        outs.append(self._forward_head(p5, self.head5, self.num_anchors_per_level[idx + 2]))

        if self.use_p6:
            p6 = self.smooth6(self.p6_act(self.p6_bn(self.p6_down(p5))))
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

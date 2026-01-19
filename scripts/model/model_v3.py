import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import math

# ----------------- Block Definitions -----------------

def init_detect_bias(head_moduledict, num_classes, p_init=0.01):
    """Initierar bias så att starten blir stabil (undviker loss explosion)."""
    bias_value = -math.log((1 - p_init) / p_init)
    with torch.no_grad():
        head_moduledict["out"]["cls"].bias.fill_(bias_value)
        # Initiera Aux-huvudet om det finns (för hybrid training)
        if "cls_aux" in head_moduledict["out"]:
            head_moduledict["out"]["cls_aux"].bias.fill_(bias_value)
        head_moduledict["out"]["box"].bias.zero_()

def conv_block(c_in, c_out, n=1):
    layers = []
    for i in range(n):
        layers.append(nn.Conv2d(c_in if i == 0 else c_out, c_out, 3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(c_out))
        layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)

class DWConvBlock(nn.Module):
    def __init__(self, c_in, c_out, n=1):
        super().__init__()
        layers = []
        for i in range(n):
            layers.extend([
                nn.Conv2d(c_in if i==0 else c_out, c_in if i==0 else c_out,
                          kernel_size=3, padding=1, groups=(c_in if i==0 else c_out), bias=False),
                nn.Conv2d(c_in if i==0 else c_out, c_out, kernel_size=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True)
            ])
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)

def make_head(A, head_depth, C, fpn_channels):
    """
    Head med dubbla klass-grenar för Hybrid Training (Main + Aux).
    """
    trunk = nn.Sequential(*[
        DWConvBlock(fpn_channels, fpn_channels) for _ in range(head_depth)
    ])

    out_layers = nn.ModuleDict({
        "box": nn.Conv2d(fpn_channels, A * 4, 1),      # Delad box-regressor
        "cls": nn.Conv2d(fpn_channels, A * C, 1),      # Main (One-to-One) -> Inference
        "cls_aux": nn.Conv2d(fpn_channels, A * C, 1)   # Aux (One-to-Many) -> Training only
    })
    return nn.ModuleDict({"trunk": trunk, "out": out_layers})


def _flatten_level_outputs(outs, export_concat: bool):
    """
    Plattar ut resultaten.
    Om training=True: returnerar (concat_main, concat_aux)
    Om training=False: returnerar concat_main
    """
    flat_main = []
    flat_aux = []
    
    # Kolla om vi har aux-data i outs (det är tuples om vi tränar)
    has_aux = (len(outs) > 0 and isinstance(outs[0], tuple))

    for item in outs:
        if has_aux:
            p_main, p_aux = item
        else:
            p_main = item
            p_aux = None
            
        B, A, S, _, E = p_main.shape
        flat_main.append(p_main.view(B, -1, E))
        
        if p_aux is not None:
            flat_aux.append(p_aux.view(B, -1, E))

    cat_main = torch.cat(flat_main, dim=1)
    
    # Vid inference eller export skickar vi bara Main
    if not has_aux or export_concat:
        return cat_main
    
    # Vid träning skickar vi båda
    cat_aux = torch.cat(flat_aux, dim=1)
    return cat_main, cat_aux


def _pick_out_indices(feature_info, take: int = 3):
    n = len(feature_info)
    out_idx = list(range(n - take, n))
    reductions = [feature_info[i]["reduction"] for i in out_idx]
    chs = [feature_info[i]["num_chs"] for i in out_idx]
    return out_idx, reductions, chs


# ----------------- Main Model Class -----------------

class YOLOLiteMS(nn.Module):
    def __init__(self, 
                 backbone="resnet18", 
                 num_classes=3, 
                 img_size=640,          # <-- VIKTIG: Behövs för decoding
                 fpn_channels=128,
                 num_anchors_per_level=(1, 1, 1, 1), # Ska vara 1 för NMS-fri
                 pretrained=True,
                 depth_multiple=1.0, 
                 width_multiple=1.0, 
                 head_depth=1,
                 use_p6=True, 
                 use_p2=False):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes

        # Backbone Setup
        tmp = timm.create_model(backbone, features_only=True, pretrained=pretrained)
        take = 4 if use_p2 else 3
        out_idx, reductions, chs = _pick_out_indices(tmp.feature_info, take=take)
        self.backbone = timm.create_model(backbone, features_only=True, pretrained=pretrained, out_indices=out_idx)
        self.reductions = reductions
        self.use_p6 = use_p6
        self.use_p2 = use_p2

        # FPN Channels
        fpn_channels = int(fpn_channels * width_multiple)
        d = max(1, round(2 * depth_multiple))

        # Channel mapping
        if self.use_p2: c2, c3, c4, c5 = chs
        else: c3, c4, c5 = chs

        # Lateral Layers
        if self.use_p2: self.lateral2 = nn.Conv2d(c2, fpn_channels, 1)
        self.lateral3 = nn.Conv2d(c3, fpn_channels, 1)
        self.lateral4 = nn.Conv2d(c4, fpn_channels, 1)
        self.lateral5 = nn.Conv2d(c5, fpn_channels, 1)

        # Smooth Layers
        if self.use_p2: self.smooth2 = conv_block(fpn_channels, fpn_channels, n=d)
        self.smooth3 = conv_block(fpn_channels, fpn_channels, n=d)
        self.smooth4 = conv_block(fpn_channels, fpn_channels, n=d)
        self.smooth5 = conv_block(fpn_channels, fpn_channels, n=d)

        # P6 Layers
        self.p6_down = nn.Conv2d(fpn_channels, fpn_channels, 3, 2, 1, bias=False)
        self.p6_bn = nn.BatchNorm2d(fpn_channels)
        self.p6_act = nn.SiLU(inplace=True)
        self.smooth6 = conv_block(fpn_channels, fpn_channels, n=d)

        # Anchor Setup
        level_names = (["p2"] if self.use_p2 else []) + ["p3", "p4", "p5"] + (["p6"] if self.use_p6 else [])
        if len(num_anchors_per_level) >= 3:
            A3, A4, A5 = map(int, num_anchors_per_level[:3])
            A2, A6 = A3, A5
        else:
            A2 = A3 = A4 = A5 = A6 = int(num_anchors_per_level[0]) if len(num_anchors_per_level) else 1
        anchors_map = {"p2": A2, "p3": A3, "p4": A4, "p5": A5, "p6": A6}
        self.num_anchors_per_level = tuple(anchors_map[n] for n in level_names)

        self.export_concat = False

        # Heads
        if self.use_p2:
            self.head2 = make_head(anchors_map["p2"], head_depth, num_classes, fpn_channels)
            init_detect_bias(self.head2, num_classes)
        self.head3 = make_head(anchors_map["p3"], head_depth, num_classes, fpn_channels)
        self.head4 = make_head(anchors_map["p4"], head_depth, num_classes, fpn_channels)
        self.head5 = make_head(anchors_map["p5"], head_depth, num_classes, fpn_channels)
        init_detect_bias(self.head3, num_classes)
        init_detect_bias(self.head4, num_classes)
        init_detect_bias(self.head5, num_classes)
        if self.use_p6:
            self.head6 = make_head(anchors_map["p6"], head_depth, num_classes, fpn_channels)
            init_detect_bias(self.head6, num_classes)

        base = list(self.reductions)
        self.fpn_strides = base + ([base[-1] * 2] if self.use_p6 else [])

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.shape[-2:], mode="nearest") + y

    def _forward_head(self, p, head_dict, A):
        # 1. Feature extraction
        p = head_dict["trunk"](p)
        
        # 2. Raw predictions
        box_raw = head_dict["out"]["box"](p)
        cls_raw = head_dict["out"]["cls"](p)
        
        B, _, S, _ = box_raw.shape
        stride = self.img_size / S
        
        # 3. Internal Decoding (Raw -> Pixels xywh)
        # Detta gör att loss-funktionen blir "plug and play" utan att veta om strides.
        
        # Grid generation
        dy, dx = torch.meshgrid(torch.arange(S, device=p.device), torch.arange(S, device=p.device), indexing='ij')
        
        # Reshape [B, A*4, S, S] -> [B, A, 4, S, S]
        box_raw = box_raw.view(B, A, 4, S, S)
        tx, ty, tw, th = box_raw[:, :, 0], box_raw[:, :, 1], box_raw[:, :, 2], box_raw[:, :, 3]

        # V8-style decoding
        px = ((tx.sigmoid() * 2 - 0.5) + dx) * stride
        py = ((ty.sigmoid() * 2 - 0.5) + dy) * stride
        pw = (tw.sigmoid() * 2)**2 * stride
        ph = (th.sigmoid() * 2)**2 * stride
        
        # [B, A, 4, S, S]
        box_decoded = torch.stack([px, py, pw, ph], dim=2)

        # Main Output: [B, A, 4+C, S, S] -> Permute -> [B, A, S, S, 4+C]
        cls_raw = cls_raw.view(B, A, self.num_classes, S, S)
        out_main = torch.cat([box_decoded, cls_raw], dim=2).permute(0, 1, 3, 4, 2).contiguous()

        # 4. Aux Output (Training Only)
        if self.training and "cls_aux" in head_dict["out"]:
            cls_aux = head_dict["out"]["cls_aux"](p).view(B, A, self.num_classes, S, S)
            # Aux delar samma box-regressor, men har egen cls
            out_aux = torch.cat([box_decoded, cls_aux], dim=2).permute(0, 1, 3, 4, 2).contiguous()
            return (out_main, out_aux)
        else:
            return out_main

    def forward(self, x):
        feats = self.backbone(x)
        if self.use_p2: c2, c3, c4, c5 = feats
        else: c3, c4, c5 = feats

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

class YOLOLiteMS_CPU(nn.Module):
    def __init__(
        self,
        backbone="mobilenetv3_small_100",
        num_classes=3,
        img_size=640,          # <-- NY: Behövs för decoding
        fpn_channels=96,
        num_anchors_per_level=(1, 1, 1, 1), # 1 ankare för NMS-fri
        pretrained=True,
        depth_multiple=0.75,
        width_multiple=0.75,
        head_depth=1,
        use_p6=True,
        use_p2=False,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes

        # Backbone
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

        if self.use_p2: c2, c3, c4, c5 = chs
        else: c3, c4, c5 = chs

        # DW-friendly FPN (Samma som förut)
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

        # Anchors
        level_names = (["p2"] if self.use_p2 else []) + ["p3","p4","p5"] + (["p6"] if self.use_p6 else [])
        if len(num_anchors_per_level) >= 3:
            A3, A4, A5 = map(int, num_anchors_per_level[:3])
            A2, A6 = A3, A5
        else:
            A2 = A3 = A4 = A5 = A6 = int(num_anchors_per_level[0]) if len(num_anchors_per_level) else 1
        anchors_map = {"p2":A2, "p3":A3, "p4":A4, "p5":A5, "p6":A6}
        self.num_anchors_per_level = tuple(anchors_map[n] for n in level_names)

        self.export_concat = False

        # Heads (Använder den uppdaterade make_head med Aux)
        if self.use_p2:
            self.head2 = make_head(anchors_map["p2"], head_depth, num_classes, fpn_channels)
            init_detect_bias(self.head2, num_classes)
        self.head3 = make_head(anchors_map["p3"], head_depth, num_classes, fpn_channels)
        self.head4 = make_head(anchors_map["p4"], head_depth, num_classes, fpn_channels)
        self.head5 = make_head(anchors_map["p5"], head_depth, num_classes, fpn_channels)
        init_detect_bias(self.head3, num_classes)
        init_detect_bias(self.head4, num_classes)
        init_detect_bias(self.head5, num_classes)
        if self.use_p6:
            self.head6 = make_head(anchors_map["p6"], head_depth, num_classes, fpn_channels)
            init_detect_bias(self.head6, num_classes)

        base = list(self.reductions)
        self.fpn_strides = base + ([base[-1]*2] if self.use_p6 else [])

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.shape[-2:], mode="nearest") + y

    def _forward_head(self, p, head_dict, A):
        """
        Uppdaterad för Hybrid Training + Intern Decoding (samma logik som YOLOLiteMS)
        """
        # 1. Features
        p = head_dict["trunk"](p)
        
        # 2. Raw Outputs
        box_raw = head_dict["out"]["box"](p)
        cls_raw = head_dict["out"]["cls"](p)
        
        B, _, S, _ = box_raw.shape
        stride = self.img_size / S

        # 3. Decode -> Pixels (xywh)
        dy, dx = torch.meshgrid(torch.arange(S, device=p.device), torch.arange(S, device=p.device), indexing='ij')
        
        box_raw = box_raw.view(B, A, 4, S, S)
        tx, ty, tw, th = box_raw[:, :, 0], box_raw[:, :, 1], box_raw[:, :, 2], box_raw[:, :, 3]

        px = ((tx.sigmoid() * 2 - 0.5) + dx) * stride
        py = ((ty.sigmoid() * 2 - 0.5) + dy) * stride
        pw = (tw.sigmoid() * 2)**2 * stride
        ph = (th.sigmoid() * 2)**2 * stride
        
        box_decoded = torch.stack([px, py, pw, ph], dim=2) # [B, A, 4, S, S]

        # 4. Main Output
        cls_raw = cls_raw.view(B, A, self.num_classes, S, S)
        out_main = torch.cat([box_decoded, cls_raw], dim=2).permute(0, 1, 3, 4, 2).contiguous()

        # 5. Aux Output (Training only)
        if self.training and "cls_aux" in head_dict["out"]:
            cls_aux = head_dict["out"]["cls_aux"](p).view(B, A, self.num_classes, S, S)
            out_aux = torch.cat([box_decoded, cls_aux], dim=2).permute(0, 1, 3, 4, 2).contiguous()
            return (out_main, out_aux)
        else:
            return out_main

    def forward(self, x):
        feats = self.backbone(x)
        if self.use_p2: c2, c3, c4, c5 = feats
        else: c3, c4, c5 = feats

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

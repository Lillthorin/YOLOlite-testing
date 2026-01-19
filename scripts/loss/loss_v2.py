import math
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------- Hjälp-funktioner ---------------------------

def _targets_to_xyxy_px(tgt: dict, W: int, H: int, device: torch.device):
    """Konverterar targets till absolut pixel-format [N, 4] xyxy."""
    boxes = None
    for k in ("boxes", "bboxes", "xyxy"):
        if k in tgt and tgt[k] is not None:
            boxes = tgt[k]
            break

    if boxes is None:
        return torch.zeros((0, 4), dtype=torch.float32, device=device)

    # to tensor
    if isinstance(boxes, torch.Tensor):
        b = boxes.detach().to(device=device, dtype=torch.float32)
    else:
        b = torch.as_tensor(boxes, dtype=torch.float32, device=device)

    if b.numel() == 0:
        return b.view(0, 4)

    # heuristics: normalized vs pixels; xyxy vs xywh
    b_min = float(b.min().item())
    b_max = float(b.max().item())

    def xywh_px_to_xyxy_px(bt: torch.Tensor) -> torch.Tensor:
        x1 = bt[:, 0] - bt[:, 2] * 0.5
        y1 = bt[:, 1] - bt[:, 3] * 0.5
        x2 = bt[:, 0] + bt[:, 2] * 0.5
        y2 = bt[:, 1] + bt[:, 3] * 0.5
        return torch.stack([x1, y1, x2, y2], dim=1)

    # normalized (0..1)
    if -1e-3 <= b_min <= 1.01 and -1e-3 <= b_max <= 1.01:
        mean_wh = float((b[:, 2] + b[:, 3]).mean().item())
        if mean_wh <= 2.01:  # xywhn
            cx = b[:, 0] * W
            cy = b[:, 1] * H
            ww = b[:, 2] * W
            hh = b[:, 3] * H
            return xywh_px_to_xyxy_px(torch.stack([cx, cy, ww, hh], dim=1))
        else:  # xyxyn
            x1 = b[:, 0] * W
            y1 = b[:, 1] * H
            x2 = b[:, 2] * W
            y2 = b[:, 3] * H
            return torch.stack([x1, y1, x2, y2], dim=1)

    # pixels: decide xyxy vs xywh
    likely_xyxy = ((b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])).float().mean().item() > 0.8
    return b if likely_xyxy else xywh_px_to_xyxy_px(b)


def xywh_to_xyxy(xywh: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = xywh.unbind(-1)
    x1 = cx - w * 0.5; y1 = cy - h * 0.5
    x2 = cx + w * 0.5; y2 = cy + h * 0.5
    return torch.stack([x1, y1, x2, y2], dim=-1)


def bbox_ciou(pred_xyxy: torch.Tensor, target_xyxy: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    # Standard CIoU implementation
    px1, py1, px2, py2 = pred_xyxy.unbind(-1)
    tx1, ty1, tx2, ty2 = target_xyxy.unbind(-1)

    pw = (px2 - px1).clamp(min=eps)
    ph = (py2 - py1).clamp(min=eps)
    tw = (tx2 - tx1).clamp(min=eps)
    th = (ty2 - ty1).clamp(min=eps)

    inter_w = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(min=0)
    inter_h = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(min=0)
    inter = inter_w * inter_h
    union = pw * ph + tw * th - inter + eps
    iou = inter / union

    pcx = (px1 + px2) * 0.5
    pcy = (py1 + py2) * 0.5
    tcx = (tx1 + tx2) * 0.5
    tcy = (ty1 + ty2) * 0.5
    center_dist = (pcx - tcx) ** 2 + (pcy - tcy) ** 2

    cw = torch.max(px2, tx2) - torch.min(px1, tx1)
    ch = torch.max(py2, ty2) - torch.min(py1, ty1)
    c2 = cw ** 2 + ch ** 2 + eps

    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(tw / th) - torch.atan(pw / ph), 2)
    with torch.no_grad():
        alpha = v / (v - iou + 1 + eps)

    ciou = iou - (center_dist / c2) - alpha * v
    return ciou


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    # Pairwise IoU
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)
    
    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])
    
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h
    union = area1[:, None] + area2 - inter + eps
    return (inter / union).clamp(min=0.0, max=1.0)

# --------------------------- End-to-End Loss ---------------------------

class LossAF(nn.Module):
    """
    NMS-fri loss (End-to-End).
    1. Tar emot lista med features från modellen.
    2. Plattar ut ALLA nivåer till [Batch, TotalAnchors, Features].
    3. Använder One-to-One matching (lägst cost vinner GT).
    4. Focal Loss för klass (objektness är inbakat i klass 0..C).
    5. CIoU för box.
    """

    def __init__(self,
                 num_classes: int,
                 img_size: int,
                 lambda_box: float = 7.5,
                 lambda_cls: float = 0.5,
                 center_mode: str = "v8",
                 wh_mode: str = "softplus",
                 # Matching parameters
                 topk_candidates: int = 10,  # Hur många ankare vi tittar på innan vi väljer DEN bästa
                 alpha_cost: float = 0.5,    # Vikt för class cost i matchningen
                 beta_cost: float = 6.0,     # Vikt för box cost i matchningen
                 # Focal Loss params
                 gamma: float = 2.0,
                 alpha: float = 0.25):
        super().__init__()
        self.num_classes = int(num_classes)
        self.img_size = int(img_size)
        
        self.lambda_box = float(lambda_box)
        self.lambda_cls = float(lambda_cls)
        
        self.center_mode = center_mode
        self.wh_mode = wh_mode
        
        self.topk_candidates = topk_candidates
        self.alpha_cost = alpha_cost
        self.beta_cost = beta_cost
        
        self.gamma = gamma
        self.alpha = alpha

    def _decode(self, p, stride):
        """Decodar råa outputs till xywh (pixels). p: [B, A, S, S, E]"""
        device = p.device
        B, A, S, _, E = p.shape
        
        # Split: box (0-3), cls (4:) -- OBS: Ingen OBJ
        tx, ty, tw, th = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
        tcls_logits = p[..., 4:]

        # Grid setup
        gy, gx = torch.meshgrid(
            torch.arange(S, device=device),
            torch.arange(S, device=device),
            indexing="ij"
        )
        gx = gx.view(1, 1, S, S)
        gy = gy.view(1, 1, S, S)

        # Center decode
        if self.center_mode == "v8":
            px = ((torch.sigmoid(tx) * 2.0 - 0.5) + gx) * stride
            py = ((torch.sigmoid(ty) * 2.0 - 0.5) + gy) * stride
        else:
            px = (torch.sigmoid(tx) + gx) * stride
            py = (torch.sigmoid(ty) + gy) * stride

        # WH decode
        if self.wh_mode == "softplus":
             pw = F.softplus(tw) * stride
             ph = F.softplus(th) * stride
        elif self.wh_mode == "v8":
             pw = (torch.sigmoid(tw) * 2).pow(2) * stride
             ph = (torch.sigmoid(th) * 2).pow(2) * stride
        else: # exp
             pw = torch.exp(tw.clamp(-10, 8)) * stride
             ph = torch.exp(th.clamp(-10, 8)) * stride

        # Output flatten: [B, N_level, 4] och [B, N_level, C]
        pred_box = torch.stack([px, py, pw, ph], dim=-1).view(B, -1, 4)
        pred_cls = tcls_logits.view(B, -1, self.num_classes)
        
        return pred_box, pred_cls

    def _get_cost(self, pred_cls, pred_box, gt_cls, gt_box, gt_labels):
        """
        Beräknar kostnadsmatris för matchning. 
        Cost = alpha * L_cls + beta * L_box
        """
        # Cls cost: Focal Loss-liknande score.
        # Vi vill ha hög score för rätt klass. Cost = -pred_prob(rätt klass)
        # pred_cls: [Np, C] (logits), gt_labels: [N]
        pred_prob = torch.sigmoid(pred_cls)
        # Hämta prob för rätt klass för varje GT
        # [Np, N] matrix
        cls_score = pred_prob[:, gt_labels] 
        
        # Box cost: CIoU (vi vill maximera IoU -> minimera 1-IoU)
        iou = bbox_iou(xywh_to_xyxy(pred_box), gt_box) # [Np, N]
        
        # Total cost (lägre är bättre)
        cost = (self.alpha_cost * (1.0 - cls_score)) + (self.beta_cost * (1.0 - iou))
        return cost

    def forward(self, preds: List[torch.Tensor], targets: List[dict]):
        device = preds[0].device
        B = preds[0].shape[0]

        # 1. Flatten all levels (Global Matching)
        all_pred_boxes = []
        all_pred_logits = []
        
        for p in preds:
            S = p.shape[2]
            stride = self.img_size / S
            pb, pl = self._decode(p, stride)
            all_pred_boxes.append(pb)
            all_pred_logits.append(pl)
            
        # Concat: [B, Total_Anchors, 4]
        flat_boxes = torch.cat(all_pred_boxes, dim=1) 
        flat_logits = torch.cat(all_pred_logits, dim=1)
        
        loss_box = torch.tensor(0.0, device=device)
        loss_cls = torch.tensor(0.0, device=device)
        total_pos = 0

        # Loop over batch
        for b in range(B):
            tgt_xyxy = _targets_to_xyxy_px(targets[b], self.img_size, self.img_size, device)
            tgt_labels = targets[b]["labels"].long().to(device)
            N = tgt_xyxy.shape[0]

            p_box = flat_boxes[b]     # [Total, 4] (xywh)
            p_logits = flat_logits[b] # [Total, C]
            
            # --- Skapa targets ---
            # Default: alla är bakgrund (target=0)
            cls_targets = torch.zeros_like(p_logits) 
            
            if N > 0:
                # Konvertera boxar för matchning
                p_xyxy = xywh_to_xyxy(p_box)
                
                # --- Pre-filtering (minska sökyta för matchning) ---
                # Hitta kandidater som är nära GT-centra (SimOTA-stil eller Center Sampling)
                # För enkelhetens skull: TopK närmaste centers
                tgt_cx = (tgt_xyxy[:, 0] + tgt_xyxy[:, 2]) / 2
                tgt_cy = (tgt_xyxy[:, 1] + tgt_xyxy[:, 3]) / 2
                
                p_cx = p_box[:, 0]
                p_cy = p_box[:, 1]
                
                # Avstånd [Total, N]
                dist = (p_cx.unsqueeze(1) - tgt_cx.unsqueeze(0))**2 + \
                       (p_cy.unsqueeze(1) - tgt_cy.unsqueeze(0))**2
                
                # Välj ut top-k kandidater per GT för att räkna noggrann cost på
                # Vi tar topk_candidates * N unika ankare totalt (approximativt)
                k = min(self.topk_candidates, p_box.shape[0])
                _, candidate_idx = torch.topk(dist, k, dim=0, largest=False) # [k, N]
                candidate_idx = candidate_idx.flatten().unique()
                
                # Extrahera kandidater
                cand_logits = p_logits[candidate_idx]
                cand_box = p_box[candidate_idx]
                
                # --- One-to-One Matching (Greedy Assignment via Cost) ---
                cost_matrix = self._get_cost(cand_logits, cand_box, tgt_xyxy, tgt_xyxy, tgt_labels) # [Cand, N]
                
                # Assign: För varje GT, hitta minsta cost
                # OBS: För strikt One-to-One får inget ankare ha två GTs.
                # En enkel lösning: min(dim=0). Om kollision, vinner den med lägst cost.
                values, indices = torch.min(cost_matrix, dim=0) # indices är relativa till candidate_idx
                
                # Konvertera tillbaka till globala index
                matched_anchor_indices = candidate_idx[indices]
                
                # --- Loss Calculation (Positives) ---
                # 1. Box Loss (endast för matchade)
                matched_pred_xyxy = p_xyxy[matched_anchor_indices]
                ciou = bbox_ciou(matched_pred_xyxy, tgt_xyxy)
                loss_box += self.lambda_box * (1.0 - ciou).sum() / max(1, N) # Normera per bildens objekt
                
                # 2. Cls Targets (Positives)
                # Sätt target=1.0 för rätt klass på matchade index
                # Vi använder F.one_hot logic manuellt
                row_idx = torch.arange(N, device=device)
                cls_targets[matched_anchor_indices, tgt_labels] = 1.0
                
                total_pos += N

            # --- Loss Calculation (Focal Loss på ALLA) ---
            # Positives drivs mot 1, alla andra (bakgrund) mot 0
            # Sigmoid Focal Loss inbyggd
            probs = torch.sigmoid(p_logits)
            ce_loss = F.binary_cross_entropy_with_logits(p_logits, cls_targets, reduction="none")
            p_t = probs * cls_targets + (1 - probs) * (1 - cls_targets)
            loss = ce_loss * ((1 - p_t) ** self.gamma)

            if self.alpha >= 0:
                alpha_t = self.alpha * cls_targets + (1 - self.alpha) * (1 - cls_targets)
                loss = alpha_t * loss

            # Normalisera cls loss med antalet targets (som DETR/YOLOv10)
            # Eller antalet ankare? Vanligtvis normera med max(1, total_pos)
            loss_cls += self.lambda_cls * loss.sum() / max(1, N)

        # Medelvärdesbilda över batch
        return (loss_box + loss_cls) / max(1, B), {
            "box": float(loss_box) / max(1, B),
            "cls": float(loss_cls) / max(1, B),
            "pos": total_pos / max(1, B)
        }

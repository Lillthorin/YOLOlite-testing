import math
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------- Matchers ---------------------------

@torch.no_grad()
def one_to_one_greedy(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Greedy one-to-one assignment.
    En anchor kan matcha max 1 GT och vice versa.
    """
    Cand, Ngt = cost.shape
    if Cand == 0 or Ngt == 0:
        return torch.empty((0,), device=cost.device, dtype=torch.long), \
               torch.empty((0,), device=cost.device, dtype=torch.long)

    assigned = torch.full((Ngt,), -1, device=cost.device, dtype=torch.long)
    used = torch.zeros((Cand,), device=cost.device, dtype=torch.bool)

    flat = cost.reshape(-1)
    order = torch.argsort(flat)

    for idx in order:
        c = idx // Ngt
        g = idx % Ngt
        if assigned[g] != -1 or used[c]:
            continue
        assigned[g] = c
        used[c] = True
        if (assigned != -1).all():
            break

    valid_g = torch.nonzero(assigned != -1).squeeze(1)
    if valid_g.numel() == 0:
        return torch.empty((0,), device=cost.device, dtype=torch.long), \
               torch.empty((0,), device=cost.device, dtype=torch.long)

    cand_local = assigned[valid_g]
    return cand_local, valid_g


# --------------------------- Helpers ---------------------------

def xywh_to_xyxy(xywh: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = xywh.unbind(-1)
    x1, y1 = cx - 0.5 * w, cy - 0.5 * h
    x2, y2 = cx + 0.5 * w, cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)
    inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter + eps
    return (inter / union).clamp(0, 1)

def bbox_ciou(pred_xyxy: torch.Tensor, target_xyxy: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    px1, py1, px2, py2 = pred_xyxy.unbind(-1)
    tx1, ty1, tx2, ty2 = target_xyxy.unbind(-1)
    pw = (px2 - px1).clamp(min=eps); ph = (py2 - py1).clamp(min=eps)
    tw = (tx2 - tx1).clamp(min=eps); th = (ty2 - ty1).clamp(min=eps)
    inter_w = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(min=0)
    inter_h = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(min=0)
    inter = inter_w * inter_h
    union = pw * ph + tw * th - inter + eps
    iou = inter / union
    cw = torch.max(px2, tx2) - torch.min(px1, tx1)
    ch = torch.max(py2, ty2) - torch.min(py1, ty1)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((px1+px2 - tx1-tx2)**2 + (py1+py2 - ty1-ty2)**2) / 4
    v = (4 / (math.pi ** 2)) * (torch.atan(tw / th) - torch.atan(pw / ph)).pow(2)
    with torch.no_grad(): alpha = v / (v - iou + 1.0 + eps)
    return iou - (rho2 / c2) - alpha * v

def _targets_to_xyxy_px(tgt: dict, W: int, H: int, device: torch.device) -> torch.Tensor:
    boxes = None
    for k in ("boxes", "bboxes", "xyxy"):
        if k in tgt and tgt[k] is not None: boxes = tgt[k]; break
    if boxes is None: return torch.zeros((0, 4), dtype=torch.float32, device=device)
    b = torch.as_tensor(boxes, dtype=torch.float32, device=device)
    if b.numel() == 0: return b.view(0, 4)
    if b.max() <= 1.01: b = b * W 
    return b 

def sigmoid_focal_loss(logits, targets, gamma=2.0, alpha=0.25, reduction="none"):
    probs = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    loss = ce * ((1.0 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    if reduction == "sum": return loss.sum()
    return loss

# --------------------------- Main Loss Class ---------------------------

class LossAF(nn.Module):
    """
    Samma Loss som gav 67% mAP, med fix för class_weights krasch.
    """
    def __init__(
        self,
        num_classes: int,
        img_size: int,
        lambda_box: float = 7.5,
        lambda_cls: float = 0.5,
        center_mode: str = "v8",
        wh_mode: str = "softplus",
        topk_candidates: int = 10,
        alpha_cost: float = 0.5,
        beta_cost: float = 6.0,
        gamma: float = 2.0,
        alpha: float = 0.25,
        class_weights: Optional[List[float]] = [1.1757211179195934, 0.09527723808100434, 1.7290016439994023], # <-- Används för att rädda Referee
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.img_size = int(img_size)

        self.lambda_box = float(lambda_box)
        self.lambda_cls = float(lambda_cls)

        self.center_mode = center_mode
        self.wh_mode = wh_mode

        self.topk_candidates = int(topk_candidates)
        self.alpha_cost = float(alpha_cost)
        self.beta_cost = float(beta_cost)

        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.eps = float(eps)

        if class_weights is not None:
            cw = torch.as_tensor(class_weights, dtype=torch.float32).view(1, -1)
            self.register_buffer("class_weights", cw)
        else:
            self.class_weights = None

    def _decode(self, p: torch.Tensor, stride: float) -> Tuple[torch.Tensor, torch.Tensor]:
        device = p.device
        B, A, S, _, E = p.shape

        tx, ty, tw, th = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
        tcls_logits = p[..., 4:]

        gy, gx = torch.meshgrid(
            torch.arange(S, device=device),
            torch.arange(S, device=device),
            indexing="ij",
        )
        gx = gx.view(1, 1, S, S)
        gy = gy.view(1, 1, S, S)

        if self.center_mode == "v8":
            px = ((torch.sigmoid(tx) * 2.0 - 0.5) + gx) * stride
            py = ((torch.sigmoid(ty) * 2.0 - 0.5) + gy) * stride
        else:
            px = (torch.sigmoid(tx) + gx) * stride
            py = (torch.sigmoid(ty) + gy) * stride

        if self.wh_mode == "softplus":
            pw = F.softplus(tw) * stride
            ph = F.softplus(th) * stride
        elif self.wh_mode == "v8":
            pw = (torch.sigmoid(tw) * 2.0).pow(2) * stride
            ph = (torch.sigmoid(th) * 2.0).pow(2) * stride
        else:  # exp
            pw = torch.exp(tw.clamp(-10, 8)) * stride
            ph = torch.exp(th.clamp(-10, 8)) * stride

        pred_box = torch.stack([px, py, pw, ph], dim=-1).view(B, -1, 4)
        pred_cls = tcls_logits.view(B, -1, self.num_classes)
        return pred_box, pred_cls

    def _get_cost(self, pred_cls_logits, pred_box_xywh, gt_box_xyxy, gt_labels):
        p = torch.sigmoid(pred_cls_logits)
        p_sel = p[:, gt_labels].clamp(self.eps, 1.0 - self.eps)
        cost_cls = -torch.log(p_sel)
        iou = bbox_iou(xywh_to_xyxy(pred_box_xywh), gt_box_xyxy)
        cost_box = 1.0 - iou
        return (self.alpha_cost * cost_cls) + (self.beta_cost * cost_box)

    def forward(self, preds: List[torch.Tensor], targets: List[dict]) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Hantera om modellen returnerar tuple (om du råkade ha kvar Hybrid-modellen)
        if isinstance(preds, tuple):
            preds = preds[0] # Vi ignorerar Aux om det finns
            
        # Hantera om det är en lista eller redan plattat
        if not isinstance(preds, (list, tuple)):
            preds = [preds] # Gör till lista för loopen nedan (eller hantera platt direkt)
        
        # Om modellen returnerar redan plattat [B, Total, 4+C] (från model_v2 forward)
        # så måste vi hantera det. Men din model_v2 i förra inlägget returnerade platt.
        # LossAF var skriven för lista av nivåer. Låt oss anpassa för din model_v2 output!
        
        # Pga din model_v2 returnerar en enda stor tensor [B, Total, 4+C] (ej lista),
        # och den är INTE decodad till pixlar i din model_v2 (du returnerar raw output + view),
        # Vi måste vara försiktiga.
        
        # I din senaste model_v2: 
        # return _flatten_level_outputs(outs) -> [B, Total, 4+C] RAW LOGITS.
        # LossAF 'forward' förväntar sig en LISTA av nivåer [B, A, S, S, ...].
        
        # FIX: Vi måste skriva om början av forward för att ta emot den platta rå-tensorn
        # OCH vi måste veta strides för att decoda den. Det är svårt med platt tensor.
        
        # FÖRSLAG: Ändra din model_v2 att returnera LISTAN 'outs' istället för '_flatten_level_outputs'
        # ELLER så decodar vi i modellen.
        
        # Låt oss anta att du skickar in LISTAN av nivåer (ändra model_v2 -> export_concat=False).
        # Då fungerar koden nedan:
        
        device = preds[0].device
        B = preds[0].shape[0]

        all_pred_boxes = []
        all_pred_logits = []
        
        # Loopa nivåer
        for p in preds:
            # p: [B, A, S, S, 4+C] (eller [B, A*S*S, 4+C] om plattad per nivå?)
            # Din model_v2 permutar till [B, A, S, S, 4+C] -> OK.
            if p.dim() == 3: # Om flattenats per nivå
                 # Vi kan inte veta S här enkelt...
                 pass 
            
            S = p.shape[2]
            stride = self.img_size / S
            pb, pl = self._decode(p, stride)
            all_pred_boxes.append(pb)
            all_pred_logits.append(pl)

        flat_boxes = torch.cat(all_pred_boxes, dim=1)   # [B, Total, 4] (xywh)
        flat_logits = torch.cat(all_pred_logits, dim=1) # [B, Total, C]

        loss_box_sum = torch.zeros((), device=device)
        loss_cls_sum = torch.zeros((), device=device)
        total_pos = 0

        for b in range(B):
            tgt_xyxy = _targets_to_xyxy_px(targets[b], self.img_size, self.img_size, device)
            tgt_labels = targets[b].get("labels", torch.zeros((0,), dtype=torch.long)).long().to(device)
            N = int(tgt_xyxy.shape[0])

            p_box = flat_boxes[b]      # [Total, 4]
            p_logits = flat_logits[b]  # [Total, C]

            cls_targets = torch.zeros_like(p_logits)

            if N > 0:
                p_xyxy = xywh_to_xyxy(p_box)
                tgt_cx = (tgt_xyxy[:, 0] + tgt_xyxy[:, 2]) * 0.5
                tgt_cy = (tgt_xyxy[:, 1] + tgt_xyxy[:, 3]) * 0.5
                p_cx = p_box[:, 0]; p_cy = p_box[:, 1]

                dist = (p_cx[:, None] - tgt_cx[None, :]).pow(2) + (p_cy[:, None] - tgt_cy[None, :]).pow(2)
                k = min(self.topk_candidates, p_box.shape[0])
                _, candidate_idx = torch.topk(dist, k, dim=0, largest=False)
                candidate_idx = candidate_idx.flatten().unique()

                cand_logits = p_logits[candidate_idx]
                cand_box = p_box[candidate_idx]

                cost_matrix = self._get_cost(cand_logits, cand_box, tgt_xyxy, tgt_labels)
                cand_local, valid_g = one_to_one_greedy(cost_matrix)

                if valid_g.numel() > 0:
                    matched_anchor_indices = candidate_idx[cand_local]
                    tgt_xyxy_m = tgt_xyxy[valid_g]
                    tgt_labels_m = tgt_labels[valid_g]

                    matched_pred_xyxy = p_xyxy[matched_anchor_indices]
                    ciou = bbox_ciou(matched_pred_xyxy, tgt_xyxy_m)
                    loss_box_sum = loss_box_sum + (1.0 - ciou).sum()

                    cls_targets[matched_anchor_indices, tgt_labels_m] = 1.0
                    total_pos += int(valid_g.numel())

            # Focal cls loss
            loss_map = sigmoid_focal_loss(p_logits, cls_targets, gamma=self.gamma, alpha=self.alpha, reduction="none")
            
            # --- HÄR ÄR FIXEN ---
            if self.class_weights is not None:
                loss_map = loss_map * self.class_weights.to(device) # <--- .to(device)
            
            loss_cls_sum = loss_cls_sum + loss_map.sum()

        norm = float(max(1, total_pos))
        loss_box = self.lambda_box * loss_box_sum / norm
        loss_cls = self.lambda_cls * loss_cls_sum / norm
        loss = (loss_box + loss_cls) / float(max(1, B))

        return loss, {
            "box": float(loss_box.item()),
            "cls": float(loss_cls.item()),
            "pos": float(total_pos) / float(max(1, B)),
        }

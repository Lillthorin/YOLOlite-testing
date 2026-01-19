import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------- Hjälp-funktioner ---------------------------

def _targets_to_xyxy_px(tgt: dict, W: int, H: int, device: torch.device):
    boxes = None
    for k in ("boxes", "bboxes", "xyxy"):
        if k in tgt and tgt[k] is not None:
            boxes = tgt[k]
            break
    if boxes is None:
        return torch.zeros((0, 4), dtype=torch.float32, device=device)

    if isinstance(boxes, torch.Tensor):
        b = boxes.detach().to(device=device, dtype=torch.float32)
    else:
        b = torch.as_tensor(boxes, dtype=torch.float32, device=device)

    if b.numel() == 0:
        return b.view(0, 4)

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

    likely_xyxy = ((b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])).float().mean().item() > 0.8
    return b if likely_xyxy else xywh_px_to_xyxy_px(b)


def xywh_to_xyxy(xywh: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = xywh.unbind(-1)
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    return torch.stack([x1, y1, x2, y2], dim=-1)


def bbox_ciou(pred_xyxy: torch.Tensor, target_xyxy: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
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

    v = (4.0 / (math.pi ** 2)) * torch.pow(torch.atan(tw / th) - torch.atan(pw / ph), 2)
    with torch.no_grad():
        alpha = v / (v - iou + 1 + eps)

    return iou - (center_dist / c2) - alpha * v


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
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


@torch.no_grad()
def one_to_one_greedy(cost: torch.Tensor):
    """
    Riktig one-to-one (snabb greedy).
    cost: [Cand, Ngt]
    return: (cand_local_idx[K], gt_idx[K])
    """
    Cand, Ngt = cost.shape
    assigned = torch.full((Ngt,), -1, device=cost.device, dtype=torch.long)
    used = torch.zeros((Cand,), device=cost.device, dtype=torch.bool)

    flat = cost.view(-1)
    order = torch.argsort(flat)  # låg kost först

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
        return (
            torch.empty((0,), device=cost.device, dtype=torch.long),
            torch.empty((0,), device=cost.device, dtype=torch.long),
        )
    cand_local = assigned[valid_g]
    return cand_local, valid_g


def focal_loss_map(logits: torch.Tensor, targets: torch.Tensor, gamma: float, alpha: float):
    """Return [N, C] focal loss map (no reduction)."""
    probs = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    loss = ce * ((1.0 - p_t) ** gamma)

    if alpha is not None and alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    return loss


# --------------------------- LossAF (snabb + bättre) ---------------------------

class LossAF(nn.Module):
    def __init__(
        self,
        num_classes: int,
        img_size: int,
        lambda_box: float = 7.5,
        lambda_cls: float = 0.5,
        center_mode: str = "v8",
        wh_mode: str = "softplus",
        # Matching
        topk_candidates: int = 10,
        alpha_cost: float = 1.5,
        beta_cost: float = 6.0,
        use_log_cost: bool = True,
        # Focal
        gamma: float = 2.0,
        alpha: float = 0.25,
        # Optional per-class weights
        class_weights=None,  # e.g. [1.0, 2.0, 6.0] or None
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
        self.use_log_cost = bool(use_log_cost)
        self.gamma = float(gamma)
        self.alpha = float(alpha)

        if class_weights is not None:
            cw = torch.as_tensor(class_weights, dtype=torch.float32).view(1, -1)
            if cw.numel() != self.num_classes:
                raise ValueError(f"class_weights must have length {self.num_classes}, got {cw.numel()}")
            self.register_buffer("class_weights", cw)
        else:
            self.class_weights = None

    def _decode(self, p, stride):
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
            pw = (torch.sigmoid(tw) * 2).pow(2) * stride
            ph = (torch.sigmoid(th) * 2).pow(2) * stride
        else:
            pw = torch.exp(tw.clamp(-10, 8)) * stride
            ph = torch.exp(th.clamp(-10, 8)) * stride

        pred_box = torch.stack([px, py, pw, ph], dim=-1).view(B, -1, 4)
        pred_cls = tcls_logits.view(B, -1, self.num_classes)
        return pred_box, pred_cls

    def _get_cost(self, pred_logits: torch.Tensor, pred_box_xywh: torch.Tensor, gt_box_xyxy: torch.Tensor, gt_labels: torch.Tensor):
        """
        cost: [Cand, Ngt]
        """
        sel_logits = pred_logits[:, gt_labels]  # [Cand, Ngt]

        if self.use_log_cost:
            p = torch.sigmoid(sel_logits).clamp(1e-6, 1.0 - 1e-6)
            cost_cls = -torch.log(p)  # starkare signal när p är låg
        else:
            cost_cls = 1.0 - torch.sigmoid(sel_logits)

        iou = bbox_iou(xywh_to_xyxy(pred_box_xywh), gt_box_xyxy)  # [Cand, Ngt]
        cost_box = 1.0 - iou

        return (self.alpha_cost * cost_cls) + (self.beta_cost * cost_box)

    def forward(self, preds: List[torch.Tensor], targets: List[dict]):
        device = preds[0].device
        B = preds[0].shape[0]

        # Flatten all levels (snabbt)
        all_pred_boxes, all_pred_logits = [], []
        for p in preds:
            S = p.shape[2]
            stride = self.img_size / S
            pb, pl = self._decode(p, stride)
            all_pred_boxes.append(pb)
            all_pred_logits.append(pl)

        flat_boxes = torch.cat(all_pred_boxes, dim=1)   # [B, Total, 4]
        flat_logits = torch.cat(all_pred_logits, dim=1) # [B, Total, C]
        Total = flat_logits.shape[1]

        loss_box = torch.zeros((), device=device)
        loss_cls = torch.zeros((), device=device)
        total_pos = 0

        for b in range(B):
            tgt_xyxy = _targets_to_xyxy_px(targets[b], self.img_size, self.img_size, device)
            tgt_labels = targets[b].get("labels", torch.zeros((0,), dtype=torch.long)).to(device).long()
            N = int(tgt_xyxy.shape[0])

            p_box = flat_boxes[b]     # [Total, 4]
            p_logits = flat_logits[b] # [Total, C]

            cls_targets = torch.zeros_like(p_logits)

            if N > 0:
                p_xyxy = xywh_to_xyxy(p_box)

                # candidate prefilter (snabb)
                tgt_cx = (tgt_xyxy[:, 0] + tgt_xyxy[:, 2]) * 0.5
                tgt_cy = (tgt_xyxy[:, 1] + tgt_xyxy[:, 3]) * 0.5
                p_cx = p_box[:, 0]
                p_cy = p_box[:, 1]

                dist = (p_cx[:, None] - tgt_cx[None, :]).pow(2) + (p_cy[:, None] - tgt_cy[None, :]).pow(2)

                k = min(self.topk_candidates, Total)
                _, candidate_idx = torch.topk(dist, k, dim=0, largest=False)  # [k, N]
                candidate_idx = candidate_idx.flatten().unique()

                cand_logits = p_logits[candidate_idx]
                cand_box = p_box[candidate_idx]

                cost = self._get_cost(cand_logits, cand_box, tgt_xyxy, tgt_labels)  # [Cand, N]
                cand_local, valid_g = one_to_one_greedy(cost)

                if valid_g.numel() > 0:
                    matched = candidate_idx[cand_local]
                    gt_xyxy_m = tgt_xyxy[valid_g]
                    gt_lbl_m = tgt_labels[valid_g]

                    ciou = bbox_ciou(p_xyxy[matched], gt_xyxy_m)
                    loss_box = loss_box + (1.0 - ciou).sum()
                    total_pos += int(valid_g.numel())

                    cls_targets[matched, gt_lbl_m] = 1.0

            # focal cls (snabb, helt vektoriserad)
            loss_map = focal_loss_map(p_logits, cls_targets, gamma=self.gamma, alpha=self.alpha)

            # per-class weighting (valfritt, device-safe)
            if self.class_weights is not None:
                cw = self.class_weights
                if cw.device != loss_map.device:
                    cw = cw.to(loss_map.device)
                loss_map = loss_map * cw

            loss_cls = loss_cls + loss_map.sum()

        # Stabil normalisering (bättre gradients än /N per bild)
        norm_pos = max(1.0, float(total_pos))
        norm_cls = norm_pos if total_pos > 0 else float(B * Total)

        loss_box = self.lambda_box * loss_box / norm_pos
        loss_cls = self.lambda_cls * loss_cls / norm_cls

        total_loss = loss_box + loss_cls
        return total_loss, {
            "box": float(loss_box.detach().item()),
            "cls": float(loss_cls.detach().item()),
            "pos": float(total_pos) / max(1.0, float(B)),
        }

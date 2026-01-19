import os
import sys
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision.ops import box_iou, nms

# ROOT fix om du kör detta som script ibland
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# =============== Hjälpare ===============
def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def denormalize(img_tensor, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
    img = img_tensor.detach().cpu().clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img

def yolo_collate(batch):
    # batch: list of (img_tensor, target_dict)
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)

def _xyxy_to_xywh(xyxy: torch.Tensor) -> torch.Tensor:
    """Tar xyxy i valfri shape. Returnerar [N,4] xywh."""
    if xyxy is None or xyxy.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32, device=xyxy.device if xyxy is not None else 'cpu')

    if xyxy.dim() == 1 and xyxy.numel() == 4:
        xyxy = xyxy.view(1, 4)
    elif xyxy.shape[-1] != 4:
        xyxy = xyxy.view(-1, 4)

    x1, y1, x2, y2 = xyxy.unbind(-1)
    w = (x2 - x1).clamp_min(0)
    h = (y2 - y1).clamp_min(0)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return torch.stack([cx, cy, w, h], dim=-1)

# =============== Decoding & Evaluering ===============

@torch.no_grad()
def _decode_batch_to_coco_dets(preds, img_size, conf_th=0.001, iou_th=0.65, add_one=True, center_mode="v8", wh_mode="softplus"):
    """
    [FIXAD] End-to-End version för COCO eval.
    Hanterar att modellen nu returnerar [Batch, TotalAnchors, 4+C] i pixlar.
    """
    
    # 1. Standardisera input (Validering ger tensor, Träning ger tuple/list)
    if isinstance(preds, (list, tuple)):
        p = preds[0] # Ta Main output
    else:
        p = preds

    # Kontrollera dimensioner
    if p.dim() != 3:
        # Fallback om något är fel, returnera tom lista
        return [[] for _ in range(len(preds) if isinstance(preds, list) else 1)]

    B, N, D = p.shape
    
    # Modellen ger redan [cx, cy, w, h] i pixlar
    # Vi behöver konvertera till xyxy för filtrering
    cx = p[..., 0]
    cy = p[..., 1]
    w  = p[..., 2]
    h  = p[..., 3]
    tcls = p[..., 4:]

    # xywh -> xyxy
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

    out_dets = [[] for _ in range(B)]

    for b in range(B):
        b_boxes = boxes_xyxy[b]      # [N, 4]
        b_logits = tcls[b]           # [N, C]
        
        # 2. Hitta bästa klass och score
        scores = b_logits.sigmoid()
        max_scores, class_ids = scores.max(dim=-1) # [N]
        
        # 3. Filtrera på confidence (COCO mAP kräver låg tröskel, t.ex. 0.001)
        mask = max_scores > conf_th
        if not mask.any():
            continue
            
        sel_boxes = b_boxes[mask]
        sel_scores = max_scores[mask]
        sel_classes = class_ids[mask]
        
        # 4. (Valfritt) Säkerhets-NMS
        # Även om modellen är NMS-fri, hjälper detta i början av träningen
        if iou_th < 1.0:
            keep = nms(sel_boxes, sel_scores, iou_th)
            sel_boxes = sel_boxes[keep]
            sel_scores = sel_scores[keep]
            sel_classes = sel_classes[keep]
            
        # 5. Formatera för COCO (xyxy -> xywh)
        # Klipp till bildens storlek
        sel_boxes[:, 0::2].clamp_(0, img_size)
        sel_boxes[:, 1::2].clamp_(0, img_size)
        
        bxywh = _xyxy_to_xywh(sel_boxes).cpu().tolist()
        sc = sel_scores.cpu().tolist()
        cc = (sel_classes + (1 if add_one else 0)).cpu().tolist() # COCO 1-based category_id
        
        dets = []
        for bx, sc_, cid in zip(bxywh, sc, cc):
            dets.append({
                "category_id": int(cid),
                "bbox": [float(v) for v in bx],
                "score": float(sc_)
            })
        out_dets[b] = dets

    return out_dets

def _coco_eval_from_lists(coco_images, coco_anns, coco_dets, iouType="bbox", num_classes=None):
    """
    Standard COCO-evaluering. Oförändrad logik, men robustare hantering av tomma inputs.
    """
    import tempfile
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not coco_dets:
        # Returnera tomma stats
        return {k: 0.0 for k in ["AP", "AP50", "AP75", "APS", "APM", "APL", "AR", "ARS", "ARM", "ARL"]}

    # Bestäm klasser om ej givet
    if num_classes is None:
        if coco_anns:
            max_cid = max(a["category_id"] for a in coco_anns)
            num_classes = int(max(1, max_cid))
        else:
            num_classes = 80 # fallback

    categories = [{"id": i, "name": str(i)} for i in range(1, num_classes + 1)]

    # Säkra tempfiler
    gt_fd, gt_path = tempfile.mkstemp(suffix=".json")
    dt_fd, dt_path = tempfile.mkstemp(suffix=".json")
    
    try:
        with os.fdopen(gt_fd, "w", encoding="utf-8") as fg:
            json.dump({
                "info": {"description": "Auto COCO GT"},
                "images": coco_images,
                "annotations": coco_anns,
                "categories": categories,
            }, fg)

        with os.fdopen(dt_fd, "w", encoding="utf-8") as fr:
            json.dump(coco_dets, fr)

        # Tysta pycocotools output lite
        # sys.stdout = open(os.devnull, 'w') 
        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.loadRes(dt_path)
        E = COCOeval(coco_gt, coco_dt, iouType=iouType)
        E.evaluate()
        E.accumulate()
        E.summarize()
        # sys.stdout = sys.__stdout__
        
        return {
            "AP":   float(E.stats[0]),
            "AP50": float(E.stats[1]),
            "AP75": float(E.stats[2]),
            "APS":  float(E.stats[3]),
            "APM":  float(E.stats[4]),
            "APL":  float(E.stats[5]),
            "AR":   float(E.stats[8]),
            "ARS":  float(E.stats[9]),
            "ARM":  float(E.stats[10]),
            "ARL":  float(E.stats[11])
        }
    except Exception as e:
        print(f"COCO eval failed: {e}")
        return {k: 0.0 for k in ["AP", "AP50", "AP75"]}
    finally:
        if os.path.exists(gt_path): os.remove(gt_path)
        if os.path.exists(dt_path): os.remove(dt_path)

def _write_json_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _append_csv(path, header: list, row: list):
    make_header = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if make_header:
            f.write(",".join(header) + "\n")
        f.write(",".join(str(x) for x in row) + "\n")

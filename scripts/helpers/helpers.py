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
def _decode_batch_to_coco_dets(preds, img_size, conf_th=0.25, iou_th=0.45, add_one=True, center_mode="v8", wh_mode="softplus"):
    """
    [ÄNDRAD] End-to-End version.
    preds: List[Tensor] där varje Tensor är [B, A, S, S, 4+C] (inget obj!)
    eller en enda concatad tensor [B, Total, 4+C].
    """
    # Hantera om input kommer som lista av nivåer eller platt tensor
    if isinstance(preds, (list, tuple)):
        # Platta ut allt först om det är uppdelat
        all_preds = []
        for p in preds:
            # p: [B, A, S, S, 4+C] -> [B, N, 4+C]
            B, A, S, _, D = p.shape
            stride = img_size / S
            
            # Decode coords direkt här för enkelhetens skull
            tx, ty, tw, th = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
            tcls = p[..., 4:] # Nu index 4 framåt är klasser!
            
            # Grid
            gy, gx = torch.meshgrid(torch.arange(S, device=p.device), torch.arange(S, device=p.device), indexing="ij")
            
            if center_mode == "v8":
                px = ((torch.sigmoid(tx) * 2 - 0.5) + gx) * stride
                py = ((torch.sigmoid(ty) * 2 - 0.5) + gy) * stride
            else:
                px = (torch.sigmoid(tx) + gx) * stride
                py = (torch.sigmoid(ty) + gy) * stride
                
            if wh_mode == "softplus":
                pw = F.softplus(tw) * stride
                ph = F.softplus(th) * stride
            elif wh_mode == "v8":
                pw = (torch.sigmoid(tw) * 2).pow(2) * stride
                ph = (torch.sigmoid(th) * 2).pow(2) * stride
            else:
                pw = torch.exp(tw.clamp(-10, 8)) * stride
                ph = torch.exp(th.clamp(-10, 8)) * stride
                
            # xywh -> xyxy
            x1 = px - pw * 0.5
            y1 = py - ph * 0.5
            x2 = px + pw * 0.5
            y2 = py + ph * 0.5
            
            # [B, N, 4] och [B, N, C]
            boxes = torch.stack([x1, y1, x2, y2], dim=-1).view(B, -1, 4)
            logits = tcls.view(B, -1, tcls.shape[-1])
            all_preds.append(torch.cat([boxes, logits], dim=-1))
            
        preds_flat = torch.cat(all_preds, dim=1) # [B, Total, 4+C]
    else:
        # Antag redan avkodad och platt (om du ändrar din modell att göra det internt)
        preds_flat = preds

    B = preds_flat.shape[0]
    out_dets = [[] for _ in range(B)]

    for b in range(B):
        p = preds_flat[b] # [Total, 4+C]
        
        boxes_xyxy = p[:, :4]
        cls_logits = p[:, 4:]
        
        # --- NMS-Fri Logik ---
        # 1. Hitta max score per ankare
        scores = cls_logits.sigmoid()
        max_scores, class_ids = scores.max(dim=1) # [Total]
        
        # 2. Hård filtrering
        mask = max_scores > conf_th
        if not mask.any():
            continue
            
        sel_boxes = boxes_xyxy[mask]
        sel_scores = max_scores[mask]
        sel_classes = class_ids[mask]
        
        # 3. (Valfritt) Säkerhets-NMS
        # Om modellen är vältränad med End-to-End loss behövs detta oftast inte,
        # eller så kan du ha en väldigt hög tröskel (0.8) för att bara ta bort extrema dubbletter.
        # Vi sätter default till "off" eller mycket hög för äkta End-to-End känsla.
        if iou_th < 1.0:
            keep = nms(sel_boxes, sel_scores, iou_th)
            sel_boxes = sel_boxes[keep]
            sel_scores = sel_scores[keep]
            sel_classes = sel_classes[keep]
            
        # Formatera för COCO
        # xyxy -> xywh
        sel_boxes[:, 0::2].clamp_(0, img_size)
        sel_boxes[:, 1::2].clamp_(0, img_size)
        
        bxywh = _xyxy_to_xywh(sel_boxes).cpu().tolist()
        sc = sel_scores.cpu().tolist()
        cc = (sel_classes + (1 if add_one else 0)).cpu().tolist() # COCO 1-based
        
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

@torch.no_grad()
def save_val_debug_anchorfree(imgs, preds, epoch, out_dir,
                              img_size=416, conf_th=0.25, iou_th=0.45,
                              mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225),
                              center_mode="v8", wh_mode="softplus"):
    """
    [ÄNDRAD] Debug-funktion för End-to-End modellen.
    Ritar boxar direkt från klass-score.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = imgs.device
    
    # 1. Gör samma decoding som förut
    # Vi kan återanvända logiken, men här implementerar vi en snabb in-place version för visualisering
    
    preds_list = preds if isinstance(preds, (list, tuple)) else [preds]
    B = preds_list[0].shape[0]

    def _denorm(img_t):
        img = img_t.detach().float().cpu()
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)
        np_img = (img.permute(1,2,0).numpy() * 255.0).clip(0,255).astype("uint8")
        return cv2.cvtColor(np.ascontiguousarray(np_img), cv2.COLOR_RGB2BGR)

    # Decode alla nivåer
    all_boxes = []
    all_scores = []
    all_classes = []

    for p in preds_list:
        # [B, A, S, S, 4+C]
        B, A, S, _, D = p.shape
        stride = img_size / S
        
        tx, ty, tw, th = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
        tcls = p[..., 4:]
        
        gy, gx = torch.meshgrid(torch.arange(S, device=device), torch.arange(S, device=device), indexing="ij")
        
        if center_mode == "v8":
            px = ((torch.sigmoid(tx) * 2 - 0.5) + gx) * stride
            py = ((torch.sigmoid(ty) * 2 - 0.5) + gy) * stride
        else:
            px = (torch.sigmoid(tx) + gx) * stride
            py = (torch.sigmoid(ty) + gy) * stride
            
        if wh_mode == "softplus":
            pw = F.softplus(tw) * stride
            ph = F.softplus(th) * stride
        elif wh_mode == "v8":
            pw = (torch.sigmoid(tw) * 2).pow(2) * stride
            ph = (torch.sigmoid(th) * 2).pow(2) * stride
        else:
            pw = torch.exp(tw.clamp(-10, 8)) * stride
            ph = torch.exp(th.clamp(-10, 8)) * stride
            
        x1 = px - pw * 0.5
        y1 = py - ph * 0.5
        x2 = px + pw * 0.5
        y2 = py + ph * 0.5
        
        boxes = torch.stack([x1, y1, x2, y2], dim=-1).view(B, -1, 4)
        
        # Max score över klasser
        scores_prob = torch.sigmoid(tcls)
        max_sc, max_idx = scores_prob.max(dim=-1)
        max_sc = max_sc.view(B, -1)
        max_idx = max_idx.view(B, -1)
        
        all_boxes.append(boxes)
        all_scores.append(max_sc)
        all_classes.append(max_idx)

    flat_boxes = torch.cat(all_boxes, dim=1)
    flat_scores = torch.cat(all_scores, dim=1)
    flat_classes = torch.cat(all_classes, dim=1)

    # Rita per bild
    for b in range(min(B, 4)): # Max 4 bilder debug
        img_np = _denorm(imgs[b])
        
        fb = flat_boxes[b]
        fs = flat_scores[b]
        fc = flat_classes[b]
        
        # Threshold
        mask = fs > conf_th
        if not mask.any():
            cv2.imwrite(os.path.join(out_dir, f"epoch_{epoch}_b{b}.jpg"), img_np)
            continue
            
        fb = fb[mask]
        fs = fs[mask]
        fc = fc[mask]
        
        # Säkerhets-NMS för visualisering (om man vill ha renare bilder)
        if iou_th < 1.0:
            keep = nms(fb, fs, iou_th)
            fb = fb[keep]
            fs = fs[keep]
            fc = fc[keep]

        for box, score, cid in zip(fb, fs, fc):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{int(cid)}: {score:.2f}"
            cv2.putText(img_np, label, (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        cv2.imwrite(os.path.join(out_dir, f"epoch_{epoch}_b{b}.jpg"), img_np)

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------- Helper Functions -----------------

def xywh_to_xyxy(xywh: torch.Tensor) -> torch.Tensor:
    """Konverterar center-xywh till xyxy."""
    cx, cy, w, h = xywh.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def bbox_iou(box1, box2, eps=1e-7):
    """
    Beräknar IoU mellan box1 (N, 4) och box2 (M, 4).
    Returnerar (N, M) matris.
    """
    # box: x1, y1, x2, y2
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)

    # Area
    area1 = (b1_x2 - b1_x1).clamp(min=0) * (b1_y2 - b1_y1).clamp(min=0)
    area2 = (b2_x2 - b2_x1).clamp(min=0) * (b2_y2 - b2_y1).clamp(min=0)

    # Intersection
    inter_x1 = torch.max(b1_x1[:, None], b2_x1[None, :])
    inter_y1 = torch.max(b1_y1[:, None], b2_y1[None, :])
    inter_x2 = torch.min(b1_x2[:, None], b2_x2[None, :])
    inter_y2 = torch.min(b1_y2[:, None], b2_y2[None, :])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    union = area1[:, None] + area2[None, :] - inter_area + eps
    return inter_area / union

def bbox_ciou(pred_xyxy, target_xyxy, eps=1e-7):
    """
    CIoU Loss mellan matchade par.
    pred_xyxy: [N, 4], target_xyxy: [N, 4]
    """
    px1, py1, px2, py2 = pred_xyxy.unbind(-1)
    tx1, ty1, tx2, ty2 = target_xyxy.unbind(-1)

    # Intersection
    inter_x1 = torch.max(px1, tx1); inter_y1 = torch.max(py1, ty1)
    inter_x2 = torch.min(px2, tx2); inter_y2 = torch.min(py2, ty2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Union
    w1, h1 = px2 - px1, py2 - py1
    w2, h2 = tx2 - tx1, ty2 - ty1
    union = w1*h1 + w2*h2 - inter_area + eps
    iou = inter_area / union

    # Enclosing box
    cw = torch.max(px2, tx2) - torch.min(px1, tx1)
    ch = torch.max(py2, ty2) - torch.min(py1, ty1)
    c2 = cw.pow(2) + ch.pow(2) + eps

    # Center distance
    rho2 = ((px1+px2 - tx1-tx2)**2 + (py1+py2 - ty1-ty2)**2) / 4

    # Aspect ratio
    v = (4 / (3.14159 ** 2)) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (rho2 / c2) - alpha * v
    return ciou.clamp(min=-1.0, max=1.0)

def _targets_to_xyxy_px(tgt: dict, img_size: int, device: torch.device):
    """Extraherar targets och säkerställer pixel-koordinater."""
    boxes = None
    for k in ("boxes", "bboxes", "xyxy"):
        if k in tgt and tgt[k] is not None:
            boxes = tgt[k]
            break
    
    if boxes is None:
        return torch.zeros((0, 4), dtype=torch.float32, device=device)

    b = torch.as_tensor(boxes, dtype=torch.float32, device=device)
    if b.numel() == 0:
        return b.view(0, 4)

    # Antag att targets är 0..1 (xywhn eller xyxyn)
    # Om max-värdet <= 1.01 antar vi normaliserat och skalar upp
    if b.max() <= 1.01:
        b = b * img_size

    # Heuristik: Om bredd/höjd är små jämfört med x/y är det nog xyxy
    # Men låt oss anta XYXY för enkelhetens skull eller konvertera om det är XYWH
    # Här gör vi en säker konvertering om det verkar vara xywh
    # (x_center < width är omöjligt för xywh om objektet är i mitten, men...)
    # Vi antar att din dataloader ger [x1, y1, x2, y2] i pixlar. 
    # Om din dataloader ger xywh, aktivera raden nedan:
    # b = xywh_to_xyxy(b) 
    
    return b

# ----------------- Hybrid Loss Class -----------------

class LossHybrid(nn.Module):
    def __init__(self, 
                 num_classes, 
                 img_size, 
                 lambda_box=7.5, 
                 lambda_cls=0.5, 
                 lambda_aux=0.25, # Hur mycket Aux påverkar (0.25 är standard)
                 topk_o2o=1,      # Main Branch: Endast bästa matchningen (NMS-fri)
                 topk_o2m=10):    # Aux Branch: De 10 bästa (Ger snabb inlärning)
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        self.lambda_aux = lambda_aux
        self.topk_o2o = topk_o2o
        self.topk_o2m = topk_o2m

    def get_assignments(self, p_box, p_logits, gt_box, gt_lbl, k_match):
        """
        SimOTA-liknande / Dynamic Soft Label Assign
        k_match avgör om det är One-to-One eller One-to-Many.
        """
        N = gt_box.shape[0]
        Total = p_box.shape[0]
        
        if N == 0:
            return None, None

        # 1. Grovfiltrering: Hitta ankare vars center är någorlunda nära GT
        # Detta sparar minne jämfört med att räkna på alla 8400 ankare
        gt_cx = (gt_box[:,0] + gt_box[:,2]) / 2
        gt_cy = (gt_box[:,1] + gt_box[:,3]) / 2
        p_cx = (p_box[:,0] + p_box[:,2]) / 2
        p_cy = (p_box[:,1] + p_box[:,3]) / 2

        # Avstånd [Total, N]
        dist = (p_cx.unsqueeze(1) - gt_cx.unsqueeze(0)).pow(2) + \
               (p_cy.unsqueeze(1) - gt_cy.unsqueeze(0)).pow(2)
        
        # Välj top-k närmaste (geometriskt) kandidater att räkna exakt cost på
        # Vi tar 50 kandidater per GT för att vara säkra
        n_cand = min(50, Total)
        _, cand_idx = dist.topk(n_cand, dim=0, largest=False)
        cand_idx = cand_idx.flatten().unique()
        
        c_box = p_box[cand_idx]        # [Cand, 4]
        c_logits = p_logits[cand_idx]  # [Cand, C]

        # 2. Beräkna Cost Matrix
        # Cost = L_cls + L_iou
        
        # Cls Cost
        pred_prob = c_logits.sigmoid()
        # Hämta sannolikhet för "rätt" klass
        target_prob = pred_prob[:, gt_lbl] # [Cand, N]
        cost_cls = F.binary_cross_entropy_with_logits(c_logits[:, gt_lbl], torch.ones_like(target_prob), reduction='none')
        
        # IoU Cost
        iou = bbox_iou(c_box, gt_box) # [Cand, N]
        cost_box = 6.0 * (1.0 - iou)  # 6.0 är en standardvikt för box-cost
        
        total_cost = cost_cls + cost_box + 1e-6

        # 3. Matchning (One-to-One eller One-to-Many)
        matches = torch.zeros_like(total_cost, dtype=torch.bool)
        
        # För varje GT, välj de k bästa ankarna
        # Om k_match=1 blir det strikt One-to-One (NMS-fritt)
        values, best_idx = total_cost.topk(k_match, dim=0, largest=False) # [k, N]
        matches.scatter_(0, best_idx, True)
        
        # Hantera One-to-One krockar: Ett ankare får inte tillhöra två GTs
        if k_match == 1:
            # Om ett ankare valt flera GTs, behåll den med lägst cost
            if matches.sum(1).max() > 1:
                vals, gt_ind = total_cost.min(1)
                matches[:] = False
                matches[torch.arange(len(cand_idx)), gt_ind] = True # Enkelt val: minsta cost vinner

        # Mappa tillbaka till globala index
        valid_mask = matches.any(dim=1)
        matched_anchor_indices = cand_idx[valid_mask]
        
        # Hitta vilken GT som varje matchat ankare tillhör
        # argmax på bool-matrisen ger indexet där det är True
        assigned_gt_idx = total_cost[valid_mask].argmin(dim=1)
        
        return matched_anchor_indices, assigned_gt_idx

    def compute_branch_loss(self, p_flat, targets, k_match):
        """Räknar ut loss för en gren (Main eller Aux)."""
        device = p_flat.device
        B = p_flat.shape[0]
        
        loss_box = torch.tensor(0., device=device)
        loss_cls = torch.tensor(0., device=device)
        total_pos = 0

        # Input p_flat är [B, Total, 4+C] och är REDAN DECODAD till pixlar (xywh + logits)
        p_xywh = p_flat[..., :4]
        p_xyxy = xywh_to_xyxy(p_xywh)
        p_logits = p_flat[..., 4:]

        for b in range(B):
            # Hämta targets
            tgt_xyxy = _targets_to_xyxy_px(targets[b], self.img_size, device)
            lbl = targets[b]["labels"].long()
            N = tgt_xyxy.shape[0]

            # Cls Targets (börjar som 0 = bakgrund)
            tcls = torch.zeros_like(p_logits[b])

            if N > 0:
                # Gör assignments (vem matchar vad?)
                a_idx, gt_idx = self.get_assignments(p_xyxy[b], p_logits[b], tgt_xyxy, lbl, k_match)

                if a_idx is not None and len(a_idx) > 0:
                    # Positives
                    pos_box = p_xyxy[b][a_idx]
                    pos_gt = tgt_xyxy[gt_idx]

                    # 1. Box Loss (CIoU)
                    ciou = bbox_ciou(pos_box, pos_gt)
                    loss_box += (1.0 - ciou).sum()

                    # 2. Cls Targets
                    # Sätt target=1 för rätt klass på matchade ankare
                    tcls[a_idx, lbl[gt_idx]] = 1.0
                    
                    total_pos += len(a_idx)

            # 3. Cls Loss (BCE / Focal) på ALLA ankare (Positive + Negative)
            # Detta trycker ner bakgrunden mot 0 och positiverna mot 1
            loss_cls += F.binary_cross_entropy_with_logits(p_logits[b], tcls, reduction='sum')

        # Normalisera lossen med antal positiver (för att inte batch-storlek ska påverka)
        norm = max(1.0, float(total_pos))
        
        return (loss_box * self.lambda_box / norm), (loss_cls * self.lambda_cls / norm)

    def forward(self, preds, targets):
        # preds är antingen en tensor (val) eller tuple (train)
        if isinstance(preds, tuple):
            p_main, p_aux = preds
        else:
            p_main = preds
            p_aux = None

        # --- Main Branch (Inference-grenen) ---
        # Tränas med One-to-One (k=1) för att slippa NMS
        l_box_m, l_cls_m = self.compute_branch_loss(p_main, targets, k_match=self.topk_o2o)
        
        loss = l_box_m + l_cls_m
        loss_items = {
            "box": l_box_m.item(), 
            "cls": l_cls_m.item(), 
            "aux_box": 0.0, 
            "aux_cls": 0.0
        }

        # --- Aux Branch (Training-grenen) ---
        # Tränas med One-to-Many (k=10) för att ge bra gradientsignal
        if p_aux is not None:
            l_box_a, l_cls_a = self.compute_branch_loss(p_aux, targets, k_match=self.topk_o2m)
            
            # Lägg till Aux loss med en vikt (0.25)
            loss += self.lambda_aux * (l_box_a + l_cls_a)
            
            loss_items["aux_box"] = l_box_a.item()
            loss_items["aux_cls"] = l_cls_a.item()

        return loss, loss_items

import colorsys
from typing import Sequence
import numpy as np
import cv2

def _make_palette(n: int) -> list[tuple[int,int,int]]:
    """
    Stabil klasspalett (HSV → BGR). Ser ut som Ultralytics/YOLO-lika färger.
    """
    if n <= 0:
        return [(0, 255, 0)]
    hues = [i / max(1, n) for i in range(n)]
    cols = []
    for h in hues:
        r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)  # mättad, ljus
        cols.append((int(b*255), int(g*255), int(r*255)))  # BGR för cv2
    return cols

def _txt_size(text: str, font_scale: float = 0.5, thickness: int = 1):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    return w, h + base

def draw_det(img_bgr: np.ndarray,
             boxes: np.ndarray,
             scores: np.ndarray,
             classes: np.ndarray,
             names: Sequence[str]) -> np.ndarray:
    """
    YOLO-lik overlay: klassfärger, fylld labelbakgrund, tunn AA-ram.
    """
    out = img_bgr.copy()
    H, W = out.shape[:2]
    n_classes = max(len(names), int(classes.max()+1) if classes.size else 0)
    palette = _make_palette(n_classes)

    # Tjocklek/skalning relativt bildstorlek
    t = max(1, int(round(0.002 * (H + W))))         # linjetjocklek
    fs = max(0.2, 0.0009 * (H + W))                 # font-scale

    for b, s, c in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, b.tolist())
        cid = int(c)
        cls_name = names[cid] if 0 <= cid < len(names) else str(cid)
        color = palette[cid % len(palette)]

        # Box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, t, lineType=cv2.LINE_AA)

        # Label text
        label = f"{cls_name} {s:.2f}"
        tw, th = _txt_size(label, font_scale=fs, thickness=max(1, t-1))

        # Bakgrundsremsa ovanför boxen (eller under om det är tight)
        bx1, by1 = x1, y1 - th - 3
        if by1 < 0:
            by1 = y1 + th + 3
        bx2, by2 = x1 + tw + 6, by1 + th + 2

        # Fylld label-bakgrund i klassfärg
        cv2.rectangle(out, (bx1, by1), (bx2, by2), color, -1, cv2.LINE_AA)

        # Text i svart eller vit beroende på bakgrundsluminans
        luminance = 0.299*color[2] + 0.587*color[1] + 0.114*color[0]
        txt_color = (0, 0, 0) if luminance > 150 else (255, 255, 255)

        cv2.putText(out, label, (bx1 + 3, by2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, txt_color, max(1, t-1), cv2.LINE_AA)

    return out
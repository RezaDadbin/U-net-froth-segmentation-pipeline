from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_EXTS = {".jpg",".jpeg",".png",".tif",".tiff"}

def _read_rgb(p: Path) -> np.ndarray:
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(p)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def _mask_from_labelme(js: dict, H: int, W: int) -> np.ndarray:
    polys = []
    for sh in js.get("shapes", []):
        pts = np.array(sh.get("points", []), dtype=np.float32)
        if pts.ndim==2 and pts.shape[0]>=3:
            polys.append(pts.astype(np.int32))
    mask = np.zeros((H,W), dtype=np.uint8)
    if polys: cv2.fillPoly(mask, polys, 1)
    return mask

def _mask_from_simple(js: dict, H: int, W: int) -> np.ndarray:
    polys = []
    for poly in js.get("polygons", []):
        pts = np.array(poly, dtype=np.float32)
        if pts.ndim==2 and pts.shape[0]>=3:
            polys.append(pts.astype(np.int32))
    mask = np.zeros((H,W), dtype=np.uint8)
    if polys: cv2.fillPoly(mask, polys, 1)
    return mask

def _mask_from_coco_like(js: dict, H: int, W: int) -> np.ndarray:
    polys = []
    for ann in js.get("annotations", []):
        seg = ann.get("segmentation", [])
        if isinstance(seg, list):
            for comp in seg:
                arr = np.array(comp, dtype=np.float32)
                if arr.size >= 6 and (arr.size % 2)==0:
                    pts = arr.reshape(-1,2)
                    if pts.shape[0]>=3:
                        polys.append(pts.astype(np.int32))
    mask = np.zeros((H,W), dtype=np.uint8)
    if polys: cv2.fillPoly(mask, polys, 1)
    return mask

def _mask_from_json(json_path: Path, H: int, W: int) -> np.ndarray:
    with open(json_path, "r", encoding="utf-8") as f:
        js = json.load(f)
    if "shapes" in js: return _mask_from_labelme(js, H, W)
    if "polygons" in js: return _mask_from_simple(js, H, W)
    if "annotations" in js: return _mask_from_coco_like(js, H, W)
    return np.zeros((H,W), dtype=np.uint8)

class PolygonDataset(Dataset):
    def __init__(self, split_dir: Path, image_size=(512,512), augment=False, skip_missing=True):
        self.root = Path(split_dir)
        self.image_size = tuple(image_size)
        self.augment = augment
        all_imgs = sorted([p for p in self.root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
        self.pairs = []
        skipped = []
        for ip in all_imgs:
            jp = ip.with_suffix(".json")
            if jp.exists():
                self.pairs.append((ip, jp))
            else:
                if skip_missing:
                    skipped.append(ip)
        if skipped:
            print(f"[warn] {split_dir}: skipping {len(skipped)} image(s) without JSON labels.")
            for s in skipped[:10]: print(f"   - {s.name}")
            if len(skipped)>10: print(f"   ... and {len(skipped)-10} more")
        if not self.pairs:
            raise RuntimeError(f"No (image,json) pairs in {split_dir}")
        self.tf = self._build_transforms()

    def _build_transforms(self):
        W,H = self.image_size
        tf = [A.Resize(H, W, interpolation=cv2.INTER_LINEAR)]
        if self.augment:
            tf += [A.HorizontalFlip(p=0.5), A.Rotate(limit=10, p=0.25)]
        tf += [A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()]
        return A.Compose(tf)

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx: int):
        ip, jp = self.pairs[idx]
        img = _read_rgb(ip)
        H,W = img.shape[:2]
        mask = _mask_from_json(jp, H, W)
        out = self.tf(image=img, mask=mask)
        image_t = out["image"]
        mask_t  = out["mask"].unsqueeze(0)
        return image_t, mask_t, str(ip)

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config import Config
from unet_froth.utils.postprocess import watershed_refine

def _iter_probmaps(folder: Path):
    IMG_EXTS = {".png",".jpg",".jpeg",".tif",".tiff"}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p

def main():
    C = Config()
    src_dir = C.PRED_MASKS_PATH
    dst_dir = C.POST_MASKS_PATH
    dst_dir.mkdir(parents=True, exist_ok=True)

    for p in tqdm(list(_iter_probmaps(src_dir)) or []):
        prob = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if prob is None:
            print(f"[warn] skip {p} (unreadable)")
            continue
        prob01 = (prob.astype(np.float32) / 255.0)
        mask = watershed_refine(prob01, binary_thr=C.pixel_threshold, min_area=10)
        mask_u8 = (mask*255).astype(np.uint8)
        outp = dst_dir / f"{p.stem.replace('_prob','')}_mask.png"
        cv2.imwrite(str(outp), mask_u8)
        print(f"[OK] {p.name} -> {outp.name}")
    print(f"[DONE] postprocessed masks -> {dst_dir}")

if __name__ == "__main__":
    main()

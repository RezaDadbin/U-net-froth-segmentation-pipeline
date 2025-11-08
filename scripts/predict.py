from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config import Config
from unet_froth.models.unet import build_unet
from unet_froth.utils.common import load_checkpoint

IMG_EXTS = {".jpg",".jpeg",".png",".tif",".tiff"}

def _read_rgb(p: Path):
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(p)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

@torch.no_grad()
def predict_image(model, device, img_rgb: np.ndarray, image_size=(512,512)) -> np.ndarray:
    W,H = image_size
    im = cv2.resize(img_rgb, (W,H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    im /= np.float32(255.0)
    mean = np.array([0.485,0.456,0.406], dtype=np.float32)
    std  = np.array([0.229,0.224,0.225], dtype=np.float32)
    im = (im - mean) / std
    im = np.transpose(im, (2,0,1))[None, ...]  # [1,3,H,W]
    t = torch.from_numpy(im).to(device=device, dtype=torch.float32)
    model = model.to(device=device, dtype=torch.float32)
    logits = model(t)
    probs = torch.sigmoid(logits)[0,0].cpu().numpy().astype(np.float32)  # [H,W]
    return probs

def _iter_images(folder: Path):
    folder = Path(folder)
    if folder.is_file():
        if folder.suffix.lower() in IMG_EXTS:
            yield folder
        return
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p

def main():
    C = Config()
    device = C.get_device()
    ckpt = C.checkpoint_dir / (C.resume_checkpoint or "model_best.pth")
    print(f"[load] {ckpt}")
    model = build_unet(in_channels=3, out_channels=1)
    load_checkpoint(model, ckpt, device)
    model.eval()

    out_dir = C.PRED_MASKS_PATH
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = C.EVAL_PATH
    for p in tqdm(list(_iter_images(eval_dir)) or []):
        img = _read_rgb(p)
        H0,W0 = img.shape[:2]
        prob = predict_image(model, device, img, image_size=C.image_size)  # [H,W] in 0..1
        # upscale to original size for saving
        prob_up = cv2.resize(prob, (W0, H0), interpolation=cv2.INTER_LINEAR)
        prob_u8 = np.clip(prob_up*255.0, 0, 255).astype(np.uint8)
        out_path = out_dir / f"{p.stem}_prob.png"
        cv2.imwrite(str(out_path), prob_u8)
        print(f"[OK] {p.name} -> {out_path.name}")

    print(f"[DONE] prob masks -> {out_dir}")

if __name__ == "__main__":
    main()

from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from unet_froth.models.unet import build_unet
from unet_froth.data import PolygonDataset
from unet_froth.utils.common import load_checkpoint
from unet_froth.utils.metrics import iou, dice_score

def main():
    C = Config()
    device = C.get_device()
    ds = PolygonDataset(C.EVAL_PATH, image_size=C.image_size, augment=False, skip_missing=True)
    loader = DataLoader(ds, batch_size=C.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_unet(in_channels=3, out_channels=1).to(device)
    ckpt = C.checkpoint_dir / (C.resume_checkpoint or "model_best.pth")
    print(f"[load] {ckpt}")
    load_checkpoint(model, ckpt, device)
    model.eval()

    bce = torch.nn.BCEWithLogitsLoss()
    vl, viou, vdice = 0.0, 0.0, 0.0
    with torch.no_grad():
        for imgs, masks, _ in tqdm(loader, desc='eval', leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            vl += float(bce(logits, masks.float()).item())
            viou  += iou(logits, masks, thr=C.pixel_threshold)
            vdice += dice_score(logits, masks, thr=C.pixel_threshold)

    n = max(1,len(loader))
    print(f"eval_loss={vl/n:.4f} | eval_iou={viou/n:.4f} | eval_dice={vdice/n:.4f}")

if __name__ == "__main__":
    main()

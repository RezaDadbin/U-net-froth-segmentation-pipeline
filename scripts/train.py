from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from unet_froth.models.unet import build_unet
from unet_froth.data import PolygonDataset
from unet_froth.utils.common import set_seed, save_checkpoint, load_checkpoint
from unet_froth.utils.metrics import iou, dice_score

def main():
    C = Config()
    device = C.get_device()
    set_seed(C.seed)
    print(f"[device] {device}")

    tr_ds = PolygonDataset(C.TRAIN_PATH, image_size=C.image_size, augment=True,  skip_missing=True)
    val_ds = PolygonDataset(C.VAL_PATH,   image_size=C.image_size, augment=False, skip_missing=True)
    tr_loader = DataLoader(tr_ds, batch_size=C.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=C.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_unet(in_channels=3, out_channels=1).to(device)
    if C.resume_checkpoint:
        ckpt_path = C.checkpoint_dir / C.resume_checkpoint
        if ckpt_path.exists():
            print(f"[resume] {ckpt_path}")
            load_checkpoint(model, ckpt_path, device)

    optimizer = optim.Adam(model.parameters(), lr=C.lr, weight_decay=C.weight_decay)
    bce = torch.nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=(device.type=='cuda'))

    best_metric = -1.0
    C.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, C.epochs+1):
        print(f"\nEpoch {epoch}/{C.epochs}")
        model.train()
        run_loss = 0.0
        for imgs, masks, _ in tqdm(tr_loader, desc='train', leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type=='cuda')):
                logits = model(imgs)
                loss = bce(logits, masks.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            run_loss += float(loss.item())
        tr_loss = run_loss / max(1, len(tr_loader))

        model.eval()
        vl, viou, vdice = 0.0, 0.0, 0.0
        with torch.no_grad():
            for imgs, masks, _ in tqdm(val_loader, desc='valid', leave=False):
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                vl += float(bce(logits, masks.float()).item())
                viou  += iou(logits, masks, thr=C.pixel_threshold)
                vdice += dice_score(logits, masks, thr=C.pixel_threshold)
        val_loss = vl/max(1,len(val_loader))
        val_iou  = viou/max(1,len(val_loader))
        val_dice = vdice/max(1,len(val_loader))

        print(f"train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | val_iou={val_iou:.4f} | val_dice={val_dice:.4f}")

        if val_iou > best_metric:
            best_metric = val_iou
            path = C.checkpoint_dir / "model_best.pth"
            save_checkpoint({"state_dict": model.state_dict()}, path)
            print(f"[save] best -> {path} (val_iou={best_metric:.4f})")

        if C.save_every and (epoch % C.save_every == 0):
            ep_path = C.checkpoint_dir / f"model_epoch_{epoch:03d}.pth"
            save_checkpoint({"state_dict": model.state_dict()}, ep_path)
            print(f"[save] {ep_path}")

    print("[done] training finished.")

if __name__ == "__main__":
    main()

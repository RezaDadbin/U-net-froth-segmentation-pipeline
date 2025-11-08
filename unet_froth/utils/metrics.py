from __future__ import annotations
import torch

def binarize(logits: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) > thr).to(logits.dtype)

@torch.no_grad()
def iou(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> float:
    preds = binarize(logits, thr)
    targets = targets.to(dtype=preds.dtype)
    inter = (preds * targets).sum(dim=(2,3))
    union = (preds + targets - preds*targets).sum(dim=(2,3)) + eps
    return float(((inter + eps)/union).mean().item())

@torch.no_grad()
def dice_score(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> float:
    preds = binarize(logits, thr)
    targets = targets.to(dtype=preds.dtype)
    inter = (preds * targets).sum(dim=(2,3))
    denom = preds.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + eps
    return float(((2*inter + eps)/denom).mean().item())

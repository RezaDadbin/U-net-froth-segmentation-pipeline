from __future__ import annotations
import numpy as np
import cv2
from typing import Tuple

def overlay_mask_on_image(rgb: np.ndarray, mask01: np.ndarray, alpha: float = 0.45,
                          color: Tuple[int,int,int]=(0,255,0)) -> np.ndarray:
    h,w,_ = rgb.shape
    mask3 = np.dstack([mask01]*3).astype(np.uint8)
    overlay = rgb.copy().astype(np.float32)
    color_arr = np.zeros_like(overlay)
    color_arr[...,0]=color[0]; color_arr[...,1]=color[1]; color_arr[...,2]=color[2]
    overlay = np.where(mask3==1, (1-alpha)*overlay + alpha*color_arr, overlay)
    return np.clip(overlay,0,255).astype(np.uint8)

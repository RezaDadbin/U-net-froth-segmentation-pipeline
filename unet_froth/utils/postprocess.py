from __future__ import annotations
import numpy as np
import cv2
from scipy import ndimage as ndi

import numpy as np
import cv2
from scipy import ndimage as ndi

def watershed_refine(prob01: np.ndarray, binary_thr: float = 0.5, min_area: int = 10) -> np.ndarray:
    bin_mask = (prob01 >= binary_thr).astype(np.uint8)
    if int(bin_mask.sum()) == 0:
        return bin_mask

    # markers for cv2.watershed
    dist = cv2.distanceTransform((bin_mask * 255).astype(np.uint8), cv2.DIST_L2, 5)
    local_max = (cv2.dilate(dist, np.ones((3,3), np.uint8)) == dist) & (bin_mask == 1)
    markers, _ = ndi.label(local_max)
    markers = markers.astype(np.int32)

    # cv2.watershed expects a 3-channel image
    rgb = np.dstack([bin_mask*255]*3).astype(np.uint8)
    cv2.watershed(rgb, markers)

    labels = markers  # negative values are boundaries
    labels[labels < 0] = 0

    out = np.zeros_like(bin_mask)
    for lab in range(1, int(labels.max())+1):
        comp = (labels == lab).astype(np.uint8)
        if int(comp.sum()) >= min_area:
            out = np.maximum(out, comp)
    return out


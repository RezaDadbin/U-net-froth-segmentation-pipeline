from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class Config:
    data_root: Path = Path("data")
    train_dir: str = "train"
    val_dir: str   = "val"
    eval_dir: str  = "eval"
    checkpoint_dir: Path = Path("checkpoints")
    outputs_root: Path   = Path("outputs")
    pred_masks_dir: str  = "pred_masks"
    postproc_masks_dir: str = "postprocessed_masks"
    image_size: tuple[int,int] = (512,512)  # (W,H)
    batch_size: int = 4
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-5
    save_every: int = 0
    pixel_threshold: float = 0.5
    device: str = "auto"   # "auto"|"cuda"|"cpu"
    seed: int = 42
    resume_checkpoint: str | None = None

    def get_device(self) -> torch.device:
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cpu")

    @property
    def TRAIN_PATH(self) -> Path: return self.data_root / self.train_dir
    @property
    def VAL_PATH(self) -> Path:   return self.data_root / self.val_dir
    @property
    def EVAL_PATH(self) -> Path:  return self.data_root / self.eval_dir
    @property
    def PRED_MASKS_PATH(self) -> Path: return self.outputs_root / self.pred_masks_dir
    @property
    def POST_MASKS_PATH(self) -> Path: return self.outputs_root / self.postproc_masks_dir

"""Standalone DINOv2 reference-embedding helper for the training worker.

This is a self-contained copy (NOT imported from the backend) so the
``watermelon_trainer`` folder can be transferred to the training computer on its
own. It MUST stay byte-for-byte compatible with the backend embedder
(``watermelon_backend/embedder.py``): same model, same preprocessing, same
L2 normalization. If the backend embedder config changes, update this too,
otherwise every freshly trained expert's compatibility thresholds will be
calibrated against embeddings the backend cannot reproduce.
"""

from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

EMBEDDER_HUB_REPO = "facebookresearch/dinov2"
EMBEDDER_MODEL_NAME = "dinov2_vits14"
EMBEDDING_DIM = 384

_PREPROCESS = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

_model = None


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_embedder():
    global _model
    if _model is not None:
        return _model
    print(f"Loading embedder {EMBEDDER_MODEL_NAME} from torch hub ({EMBEDDER_HUB_REPO})...")
    model = torch.hub.load(EMBEDDER_HUB_REPO, EMBEDDER_MODEL_NAME)
    model.eval()
    model.to(_device())
    _model = model
    return _model


@torch.no_grad()
def embed_image(path: Path) -> np.ndarray:
    model = load_embedder()
    img = Image.open(path).convert("RGB")
    x = _PREPROCESS(img).unsqueeze(0).to(_device())
    feats = model(x).squeeze(0).float().cpu().numpy().astype(np.float32)
    return feats / max(float(np.linalg.norm(feats)), 1e-8)


def embed_paths(paths: List[Path]) -> np.ndarray:
    if not paths:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    return np.stack([embed_image(Path(p)) for p in paths])


def save_reference_npz(out_path: Path, paths: List[Path]) -> int:
    embeddings = embed_paths(paths)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        image_names=np.array([Path(p).name for p in paths]),
    )
    return int(embeddings.shape[0])

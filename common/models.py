'''
Light abstraction so we can switch between OpenCLIP, BLIP-2, etc.,
without changing callers.

Usage
-----
from common.models import encoder          # singleton
vec1 = encoder.text("a red vintage car")   # list[float] of length `embed_dim`
vec2 = encoder.image(pil_image)            # list[float] of length `embed_dim`
'''

from __future__ import annotations

import os
import io
from pathlib import Path
from typing import Iterable, List, Union

import torch
import open_clip                            # pip install open_clip_torch
from PIL import Image

# ───────────────────────────────────────────────────────────── load_model (as before)
_MODEL_CACHE: dict[tuple[str, str], tuple[open_clip.model.CLIP, callable]] = {}

def load_model(name: str = "RN50", device: str = "cpu"):
    key = (name, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    
    try:
        torch_device = torch.device(device)
    except Exception:
        raise ValueError(f"Invalid device: {device!r}")
    
    VALID = set(open_clip.list_pretrained())
    if name not in VALID:
        raise ValueError(f"Unknown model {name!r}, must be one of {sorted(VALID)}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        name, pretrained="openai", device=torch_device
    )
    model.to(torch_device).eval()
    _MODEL_CACHE[key] = (model, preprocess)
    return model, preprocess


# ───────────────────────────────────────────────────────────── Encoder wrapper
class _Encoder:
    """
    Singleton helper that hides the OpenCLIP details and always returns a *list*
    so the caller can JSON-serialise it directly.

    Attributes
    ----------
    embed_dim : int
        The dimensionality of the embeddings produced by text() and image().
    """

    def __init__(self, model: str = "RN50", device: str = "cpu"):
        self.model, self.preprocess = load_model(model, device)
        self.tokenizer = open_clip.get_tokenizer(model)
        # determine embedding dimension dynamically
        if hasattr(self.model, "visual"):
            self.embed_dim = self.model.visual.output_dim
        else:
            # fallback for text-only models
            self.embed_dim = self.model.text_projection.shape[-1]

    # ------------------------------------------------------------------ helpers
    @torch.no_grad()
    def text(
        self, prompt: str | Iterable[str]
    ) -> Union[List[float], List[List[float]]]:
        """
        Encode one prompt (str) or many (Iterable[str]).

        Returns
        -------
        List[float] or List[List[float]]
            Embedding(s) of dimension `self.embed_dim`, normalized to unit length.
        """
        batched = isinstance(prompt, (list, tuple))
        prompts = prompt if batched else [prompt]

        tokens = self.tokenizer(prompts).to(self.model.device)
        feats = self.model.encode_text(tokens)
        feats /= feats.norm(dim=-1, keepdim=True)

        feats_list = feats.cpu().tolist()
        return feats_list if batched else feats_list[0]

    @torch.no_grad()
    def image(self, img) -> List[float]:
        """
        Accepts a PIL.Image, pathlib.Path, or raw bytes and returns a vector of length `self.embed_dim`.
        """
        if isinstance(img, (str, Path)):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, bytes):
            img = Image.open(io.BytesIO(img)).convert("RGB")
        elif isinstance(img, Image.Image):
            img = img.convert("RGB")
        else:
            raise TypeError("image() expects PIL.Image, path, or bytes")

        tensor = self.preprocess(img).unsqueeze(0).to(self.model.device)
        feat = self.model.encode_image(tensor)
        feat /= feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().tolist()


# ───────────────────────────────────────────────────────────── module-level singleton
# Device can be overridden with env-var so docker-compose can run `device=cuda` if needed.
encoder = _Encoder(
    model=os.getenv("MODEL", "RN50"),
    device=os.getenv("DEVICE", "cpu")
)

"""
Light abstraction so we can switch between OpenCLIP, BLIP-2, etc.,
without changing callers.

Usage
-----
from common.models import encoder          # singleton
vec1 = encoder.text("a red vintage car")   # list[float] length 768
vec2 = encoder.image(pil_image)            # list[float] length 768
"""

from __future__ import annotations

import os
import io
from pathlib import Path
from typing import Iterable

import torch
import open_clip                            # pip install open_clip_torch
from PIL import Image

# ───────────────────────────────────────────────────────────── load_model (as before)
_MODEL_CACHE: dict[tuple[str, str], tuple[open_clip.model.CLIP, callable]] = {}


def load_model(name: str = "openclip", device: str = "cpu"):
    key = (name, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if name == "openclip":
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", device=device
        )
    elif name == "blip2":
        raise NotImplementedError("Add BLIP-2 here if you need it")
    else:
        raise ValueError(f"Unknown model '{name}'")

    model.to(device).eval()
    _MODEL_CACHE[key] = (model, preprocess)
    return model, preprocess


# ───────────────────────────────────────────────────────────── Encoder wrapper
class _Encoder:
    """
    Singleton helper that hides the OpenCLIP details and always returns a *list*
    so the caller can JSON-serialise it directly.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model, self.preprocess = load_model("openclip", device)
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")

    # ------------------------------------------------------------------ helpers
    @torch.no_grad()
    def text(self, prompt: str | Iterable[str]) -> list[float] | list[list[float]]:
        """
        Encode one prompt (*str*) or many (*Iterable[str]*).
        Returns a single vector or a list of vectors, each normalised to unit length.
        """
        batched = isinstance(prompt, (list, tuple))
        prompts = prompt if batched else [prompt]

        tokens = self.tokenizer(prompts).to(self.device)
        feats = self.model.encode_text(tokens)
        feats /= feats.norm(dim=-1, keepdim=True)

        feats = feats.cpu().tolist()
        return feats if batched else feats[0]

    @torch.no_grad()
    def image(self, img) -> list[float]:
        """
        Accepts a PIL.Image, pathlib.Path, or raw bytes and returns a 768-D vector.
        """
        if isinstance(img, (str, Path)):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, bytes):
            img = Image.open(io.BytesIO(img)).convert("RGB")
        elif isinstance(img, Image.Image):
            img = img.convert("RGB")
        else:
            raise TypeError("image() expects PIL.Image, path, or bytes")

        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(tensor)
        feat /= feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().tolist()


# ───────────────────────────────────────────────────────────── module-level singleton
# Device can be overridden with env-var so docker-compose can run `device=cuda` if needed.
encoder = _Encoder(device=os.getenv("DEVICE", "cpu"))

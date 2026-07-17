from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.utils.config import get_config_value
from src.utils.vector import l2_normalize_matrix


class OpenCLIPH14Encoder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.transform = None
        self.backend = "transformers"
        self.device = "cpu"
        self.dtype = None

    def load(self) -> "OpenCLIPH14Encoder":
        if self.model is not None:
            return self
        try:
            import torch
            from PIL import Image
            from torchvision import transforms
        except ImportError as exc:
            raise RuntimeError("OpenCLIP wrapper requires torch, torchvision, and pillow.") from exc

        model_cfg = self.config["models"]["openclip_h14"]
        model_dir = Path(model_cfg["model_dir"])
        if not model_dir.exists():
            raise FileNotFoundError(f"OpenCLIP model_dir does not exist: {model_dir}")
        hf_weight_candidates = [model_dir / "pytorch_model.bin", model_dir / "model.safetensors"]
        openclip_weight_candidates = [
            model_dir / "open_clip_pytorch_model.bin",
            model_dir / "open_clip_pytorch_model.safetensors",
            model_dir / "laion2b_s32b_b79k.bin",
        ]
        if not any(path.exists() for path in hf_weight_candidates + openclip_weight_candidates):
            raise FileNotFoundError(
                "OpenCLIP ViT-H/14 weights are missing. "
                f"Found config/tokenizer in {model_dir}, but no supported weight file."
            )

        if any(path.exists() for path in hf_weight_candidates):
            try:
                from transformers import CLIPModel, CLIPTokenizerFast
            except ImportError as exc:
                raise RuntimeError("HuggingFace OpenCLIP weights require transformers.") from exc
            self.backend = "transformers"
            self.tokenizer = CLIPTokenizerFast.from_pretrained(str(model_dir), local_files_only=True)
            self.model = CLIPModel.from_pretrained(str(model_dir), local_files_only=True)
        else:
            try:
                import open_clip
            except ImportError as exc:
                raise RuntimeError("open_clip_pytorch_model.* checkpoints require open_clip_torch.") from exc
            self.backend = "open_clip"
            weight_path = next(path for path in openclip_weight_candidates if path.exists())
            model_name = str(model_cfg.get("model_name", "ViT-H-14-quickgelu"))
            self.model, _, preprocess_val = open_clip.create_model_and_transforms(model_name, pretrained=str(weight_path))
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.transform = preprocess_val

        requested_device = str(model_cfg.get("device", "cuda"))
        self.device = requested_device if torch.cuda.is_available() and requested_device.startswith("cuda") else "cpu"
        dtype_name = str(model_cfg.get("dtype", "fp16"))
        self.dtype = torch.float16 if self.device.startswith("cuda") and dtype_name == "fp16" else torch.float32
        self.model = self.model.to(self.device)
        if self.dtype == torch.float16:
            self.model = self.model.half()
        self.model.eval()

        if self.transform is None:
            preprocess_cfg = _read_openclip_preprocess(model_dir)
            mean = preprocess_cfg.get("mean", [0.48145466, 0.4578275, 0.40821073])
            std = preprocess_cfg.get("std", [0.26862954, 0.26130258, 0.27577711])
            image_size = int(get_config_value(self.config, "models.openclip_h14.image_size", 224))
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        return self

    def encode_texts(self, texts: Sequence[str], batch_size: int | None = None) -> List[List[float]]:
        self.load()
        import torch

        assert self.model is not None and self.tokenizer is not None
        batch_size = batch_size or int(get_config_value(self.config, "modal_online_sd_enabled.openclip_text_batch_size", 16))
        vectors: List[List[float]] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = list(texts[start : start + batch_size])
                if self.backend == "transformers":
                    inputs = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
                    features = self.model.get_text_features(**inputs)
                else:
                    tokens = self.tokenizer(batch_texts).to(self.device)
                    features = self.model.encode_text(tokens)
                vectors.extend(features.float().cpu().tolist())
        return l2_normalize_matrix(vectors)

    def encode_images(self, images: Sequence[Any], batch_size: int | None = None) -> List[List[float]]:
        self.load()
        import torch

        assert self.model is not None and self.transform is not None
        batch_size = batch_size or int(get_config_value(self.config, "offline.openclip_image_batch_size", 32))
        vectors: List[List[float]] = []
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                batch_images = images[start : start + batch_size]
                tensors = [self.transform(_ensure_pil_rgb(image)) for image in batch_images]
                pixel_values = torch.stack(tensors).to(self.device)
                if self.dtype == torch.float16:
                    pixel_values = pixel_values.half()
                if self.backend == "transformers":
                    features = self.model.get_image_features(pixel_values=pixel_values)
                else:
                    features = self.model.encode_image(pixel_values)
                vectors.extend(features.float().cpu().tolist())
        return l2_normalize_matrix(vectors)


def _read_openclip_preprocess(model_dir: Path) -> Dict[str, Any]:
    config_path = model_dir / "open_clip_config.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as reader:
        payload = json.load(reader)
    return dict(payload.get("preprocess_cfg") or {})


def _ensure_pil_rgb(image: Any):
    from PIL import Image

    if isinstance(image, (str, os.PathLike)):
        return Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    try:
        import numpy as np

        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
    except Exception:
        pass
    raise TypeError(f"Unsupported image type for OpenCLIP: {type(image)!r}")


def load_openclip(config: Dict[str, Any]) -> OpenCLIPH14Encoder:
    return OpenCLIPH14Encoder(config).load()

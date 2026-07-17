from __future__ import annotations

import math
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.utils.config import get_config_value
from src.utils.vector import l2_normalize_matrix


class BEiT3Encoder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.transform = None
        self.device = "cpu"
        self.dtype = None

    def load(self) -> "BEiT3Encoder":
        if self.model is not None:
            return self
        try:
            import torch
            from PIL import Image
            from torchvision import transforms
        except ImportError as exc:
            raise RuntimeError("BEiT-3 requires torch, torchvision, pillow, timm, and torchscale.") from exc

        model_cfg = self.config["models"]["beit3"]
        external_dir = Path(model_cfg["external_code_dir"]).resolve()
        if str(external_dir) not in sys.path:
            sys.path.insert(0, str(external_dir))
        if "torch._six" not in sys.modules:
            torch_six = types.ModuleType("torch._six")
            torch_six.inf = math.inf
            sys.modules["torch._six"] = torch_six

        try:
            from modeling_finetune import BEiT3ForRetrieval
            from modeling_utils import _get_large_config
            import utils as beit3_utils
        except ImportError as exc:
            raise RuntimeError(f"Cannot import BEiT-3 external code from {external_dir}.") from exc

        model_dir = Path(model_cfg["model_dir"])
        checkpoint_path = model_dir / model_cfg["checkpoint"]
        sentencepiece_path = model_dir / model_cfg["sentencepiece"]
        if not checkpoint_path.exists() or not sentencepiece_path.exists():
            raise FileNotFoundError(f"Missing BEiT-3 checkpoint or sentencepiece model under {model_dir}.")

        input_size = int(model_cfg.get("input_size", 224))
        args = _get_large_config(img_size=input_size)
        args.normalize_output = True
        model = BEiT3ForRetrieval(args)
        beit3_utils.load_model_and_may_interpolate(str(checkpoint_path), model, "model|module", "")

        requested_device = str(model_cfg.get("device", "cuda"))
        self.device = requested_device if torch.cuda.is_available() and requested_device.startswith("cuda") else "cpu"
        dtype_name = str(model_cfg.get("dtype", "fp16"))
        self.dtype = torch.float16 if self.device.startswith("cuda") and dtype_name == "fp16" else torch.float32
        model = model.to(self.device)
        if self.dtype == torch.float16:
            model = model.half()
        model.eval()

        try:
            from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
        except Exception:
            IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
            IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)

        self.transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
            ]
        )
        self.tokenizer = BEiT3SentencePieceTokenizer(sentencepiece_path)
        self.model = model
        return self

    def encode_images(self, images: Sequence[Any], batch_size: int | None = None) -> List[List[float]]:
        self.load()
        import torch

        assert self.model is not None and self.transform is not None
        batch_size = batch_size or int(get_config_value(self.config, "offline.beit3_image_batch_size", 16))
        vectors: List[List[float]] = []
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                batch_images = images[start : start + batch_size]
                tensors = [self.transform(_ensure_pil_rgb(image)) for image in batch_images]
                image_tensor = torch.stack(tensors).to(self.device)
                if self.dtype == torch.float16:
                    image_tensor = image_tensor.half()
                vision_cls, _ = self.model(image=image_tensor, only_infer=True)
                vectors.extend(vision_cls.float().cpu().tolist())
        return l2_normalize_matrix(vectors)

    def encode_texts(self, texts: Sequence[str], batch_size: int | None = None) -> List[List[float]]:
        self.load()
        import torch

        assert self.model is not None and self.tokenizer is not None
        batch_size = batch_size or int(get_config_value(self.config, "models.beit3.text_batch_size", 16))
        max_len = int(get_config_value(self.config, "models.beit3.num_max_bpe_tokens", 64))
        vectors: List[List[float]] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                token_rows = []
                mask_rows = []
                for text in texts[start : start + batch_size]:
                    tokens = self.tokenizer.tokenize(text)
                    tokens = tokens[: max_len - 2]
                    token_ids = [self.tokenizer.bos_token_id] + self.tokenizer.convert_tokens_to_ids(tokens) + [self.tokenizer.eos_token_id]
                    padding_len = max_len - len(token_ids)
                    token_rows.append(token_ids + [self.tokenizer.pad_token_id] * padding_len)
                    mask_rows.append([0] * len(token_ids) + [1] * padding_len)
                text_tensor = torch.tensor(token_rows, dtype=torch.long, device=self.device)
                padding_mask = torch.tensor(mask_rows, dtype=torch.long, device=self.device)
                _, language_cls = self.model(text_description=text_tensor, padding_mask=padding_mask, only_infer=True)
                vectors.extend(language_cls.float().cpu().tolist())
        return l2_normalize_matrix(vectors)


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
    raise TypeError(f"Unsupported image type for BEiT-3: {type(image)!r}")


class BEiT3SentencePieceTokenizer:
    """Small XLM-R compatible tokenizer wrapper for the BEiT-3 sentencepiece model."""

    bos_token_id = 0
    pad_token_id = 1
    eos_token_id = 2
    unk_token_id = 3
    fairseq_offset = 1

    _special_tokens_to_ids = {
        "<s>": bos_token_id,
        "<pad>": pad_token_id,
        "</s>": eos_token_id,
        "<unk>": unk_token_id,
    }

    def __init__(self, model_path: str | os.PathLike[str]):
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise RuntimeError("sentencepiece is required for BEiT-3 tokenization.") from exc

        self.sp_model = spm.SentencePieceProcessor()
        loaded = self.sp_model.Load(str(model_path))
        if not loaded:
            raise RuntimeError(f"Cannot load BEiT-3 sentencepiece model from {model_path}.")

    def tokenize(self, text: str) -> List[str]:
        return list(self.sp_model.EncodeAsPieces(text))

    def convert_tokens_to_ids(self, tokens: str | Sequence[str]) -> int | List[int]:
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(token) for token in tokens]

    def _convert_token_to_id(self, token: str) -> int:
        if token in self._special_tokens_to_ids:
            return self._special_tokens_to_ids[token]
        spm_id = int(self.sp_model.PieceToId(token))
        if spm_id == self.sp_model.unk_id():
            return self.unk_token_id
        return spm_id + self.fairseq_offset


def load_beit3(config: Dict[str, Any]) -> BEiT3Encoder:
    return BEiT3Encoder(config).load()

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.utils.hashing import file_sha256


class StableDiffusionGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pipeline = None
        self.device = "cpu"

    def load(self) -> "StableDiffusionGenerator":
        if self.pipeline is not None:
            return self
        try:
            import torch
            from diffusers import StableDiffusionXLPipeline
        except ImportError as exc:
            raise RuntimeError("Stable Diffusion branch requires torch and diffusers.") from exc
        model_cfg = self.config["models"]["stable_diffusion"]
        model_dir = Path(model_cfg["model_dir"])
        missing_files = [
            path
            for path in [
                model_dir / "model_index.json",
                model_dir / "unet" / "config.json",
            ]
            if not path.exists()
        ]
        if missing_files:
            formatted = ", ".join(str(path) for path in missing_files)
            raise FileNotFoundError(f"Missing SDXL diffusers model files: {formatted}.")
        requested_device = str(model_cfg.get("device", "cuda"))
        self.device = requested_device if torch.cuda.is_available() and requested_device.startswith("cuda") else "cpu"
        dtype = torch.float16 if self.device.startswith("cuda") and model_cfg.get("dtype") == "fp16" else torch.float32
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "local_files_only": True,
        }
        variant = model_cfg.get("variant") or _infer_diffusers_variant(model_dir)
        if variant:
            load_kwargs["variant"] = variant
            load_kwargs["use_safetensors"] = True
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(str(model_dir), **load_kwargs)
        self.pipeline = self.pipeline.to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)
        return self

    def build_prompt(self, query: str) -> str:
        return (
            f"A realistic video keyframe showing: {query}.\n"
            "Preserve the exact people, actions, objects, location, time, colors, and quantities from the query.\n"
            "Do not add extra objects or actions.\n"
            "Do not stylize.\n"
            "Natural video frame, realistic scene, documentary-like."
        )

    def generate(self, query: str, output_dir: str | Path) -> List[Dict[str, Any]]:
        self.load()
        import torch

        assert self.pipeline is not None
        model_cfg = self.config["models"]["stable_diffusion"]
        seeds = [int(seed) for seed in model_cfg.get("seeds", [10, 100, 1000, 10000, 100000])]
        prompt = self.build_prompt(query)
        negative_prompt = model_cfg.get("negative_prompt", "")
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        generated: List[Dict[str, Any]] = []
        for seed in seeds:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            image = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=float(model_cfg.get("guidance_scale", 7.0)),
                num_inference_steps=int(model_cfg.get("num_inference_steps", 30)),
                height=int(model_cfg.get("height", 512)),
                width=int(model_cfg.get("width", 512)),
                generator=generator,
            ).images[0]
            image_path = output_root / f"seed_{seed}.png"
            image.save(image_path)
            generated.append({"seed": seed, "path": str(image_path), "sha256": file_sha256(image_path)})
        return generated


def load_stable_diffusion(config: Dict[str, Any]) -> StableDiffusionGenerator:
    return StableDiffusionGenerator(config).load()


def _infer_diffusers_variant(model_dir: Path) -> str | None:
    fp16_files = [
        model_dir / "text_encoder" / "model.fp16.safetensors",
        model_dir / "text_encoder_2" / "model.fp16.safetensors",
        model_dir / "unet" / "diffusion_pytorch_model.fp16.safetensors",
        model_dir / "vae" / "diffusion_pytorch_model.fp16.safetensors",
    ]
    if any(path.exists() for path in fp16_files):
        return "fp16"
    return None

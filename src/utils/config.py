from __future__ import annotations

import copy
import os
import ast
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_CONFIG_PATH = "configs/default.yaml"


def _require_yaml():
    try:
        import yaml
    except ImportError as exc:
        return None
    return yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml(path: str | os.PathLike[str]) -> Dict[str, Any]:
    yaml = _require_yaml()
    if yaml is None:
        return _load_simple_yaml(path)
    with open(path, "r", encoding="utf-8") as reader:
        data = yaml.safe_load(reader) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _load_simple_yaml(path: str | os.PathLike[str]) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: List[tuple[int, Dict[str, Any]]] = [(-1, root)]
    with open(path, "r", encoding="utf-8") as reader:
        for raw_line in reader:
            if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                continue
            indent = len(raw_line) - len(raw_line.lstrip(" "))
            line = raw_line.strip()
            if ":" not in line:
                raise RuntimeError("Install PyYAML for full YAML syntax support.")
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if value == "":
                child: Dict[str, Any] = {}
                parent[key] = child
                stack.append((indent, child))
            else:
                parent[key] = _parse_simple_yaml_scalar(value)
    return root


def _parse_simple_yaml_scalar(value: str) -> Any:
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except Exception:
            inner = value[1:-1].strip()
            return [] if not inner else [item.strip().strip("'\"") for item in inner.split(",")]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_config(path: str | None = None, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    config_path = path or os.environ.get("CONFIG_PATH") or DEFAULT_CONFIG_PATH
    config = load_yaml(config_path)
    if overrides:
        config = _deep_merge(config, overrides)
    _apply_env_overrides(config)
    return config


def _apply_env_overrides(config: Dict[str, Any]) -> None:
    model_dir = os.environ.get("MODEL_DIR")
    if model_dir:
        config.setdefault("paths", {})["model_dir"] = model_dir
        models = config.setdefault("models", {})
        default_model_paths = {
            "beit3": "beit3-large-itc",
            "openclip_h14": "openclip-dfn5b-vit-h-14",
            "stable_diffusion": "sdxl-base-1.0-diffusers-fp16",
        }
        for model_key, child in default_model_paths.items():
            if model_key in models:
                models[model_key]["model_dir"] = str(Path(model_dir) / child)
        if "transnetv2" in models:
            models["transnetv2"]["weights_dir"] = str(Path(model_dir) / "transnetv2" / "transnetv2-weights")
        if "codetr" in models:
            models["codetr"]["checkpoint"] = str(Path(model_dir) / "codetr-vit-large-coco" / "pytorch_model.pth")


def get_config_value(config: Dict[str, Any], dotted_path: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def require_config_value(config: Dict[str, Any], dotted_path: str) -> Any:
    value = get_config_value(config, dotted_path)
    if value is None:
        raise KeyError(f"Missing config value: {dotted_path}")
    return value


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def validate_model_assets(config: Dict[str, Any]) -> Dict[str, List[str]]:
    missing: Dict[str, List[str]] = {}

    def add_missing(model_key: str, paths: Iterable[str]) -> None:
        for path in paths:
            if not Path(path).exists():
                missing.setdefault(model_key, []).append(path)

    beit3 = config.get("models", {}).get("beit3", {})
    beit3_dir = Path(beit3.get("model_dir", ""))
    if beit3:
        add_missing("beit3", [str(beit3_dir / beit3.get("checkpoint", "")), str(beit3_dir / beit3.get("sentencepiece", ""))])

    transnet = config.get("models", {}).get("transnetv2", {})
    if transnet:
        add_missing("transnetv2", [str(Path(transnet.get("weights_dir", "")) / "saved_model.pb")])

    codetr = config.get("models", {}).get("codetr", {})
    if codetr:
        add_missing("codetr", [str(codetr.get("checkpoint", "")), str(codetr.get("config", ""))])

    sd = config.get("models", {}).get("stable_diffusion", {})
    if sd:
        sd_dir = Path(sd.get("model_dir", ""))
        add_missing("stable_diffusion", [str(sd_dir / "model_index.json"), str(sd_dir / "unet" / "config.json")])

    openclip = config.get("models", {}).get("openclip_h14", {})
    if openclip:
        model_dir = Path(openclip.get("model_dir", ""))
        weight_candidates = [
            model_dir / "pytorch_model.bin",
            model_dir / "model.safetensors",
            model_dir / "open_clip_pytorch_model.bin",
            model_dir / "open_clip_pytorch_model.safetensors",
            model_dir / "laion2b_s32b_b79k.bin",
        ]
        if not any(path.exists() for path in weight_candidates):
            missing.setdefault("openclip_h14", []).append(
                "No OpenCLIP weight file found. Expected one of: "
                + ", ".join(str(path) for path in weight_candidates)
            )
    return missing

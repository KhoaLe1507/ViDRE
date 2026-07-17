from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from src.models.load_codetr import COCO_CLASSES
from src.utils.config import get_config_value


PARAPHRASE_PROMPT = """You are generating search-oriented paraphrases for a text-to-video/keyframe retrieval system.

Given the original query, produce exactly 5 paraphrases.

Rules:
- Preserve the original intent exactly.
- Do not add new objects, actions, people, locations, colors, quantities, or temporal relations.
- Do not remove important objects, actions, people, locations, colors, quantities, or temporal relations.
- Make the paraphrases useful for visual retrieval.
- Use concise natural English.
- Do not make the sentence more beautiful; make it more searchable.
- Return valid JSON only.
- JSON format:
{
  "paraphrases": [
    "...",
    "...",
    "...",
    "...",
    "..."
  ]
}

Original query:
{{query}}
"""


OBJECT_PROMPT = """You extract object-count constraints for a COCO-object hard filter in a text-to-keyframe retrieval system.

Given a query, extract only clear, countable visual objects that can be mapped to COCO classes.

Rules:
- Return only objects explicitly mentioned or strongly implied by the query.
- Use integer counts.
- If count is not specified but the object is clearly singular, use count = 1.
- If count is ambiguous or uncertain, omit the object.
- Do not extract actions.
- Do not extract attributes such as color unless they are part of object identity.
- Do not extract non-COCO objects unless they can be safely mapped to a COCO class.
- Return valid JSON only.

JSON format:
{
  "objects": [
    {"name": "person", "count": 1},
    {"name": "bottle", "count": 1}
  ]
}

Query:
{{query}}
"""


SCENE_REWRITE_PROMPT = """You rewrite short action queries for a text-to-keyframe retrieval system.

The retrieval system searches static video keyframes, so rewrite the query into one concise English sentence that is more visually searchable as a scene/object/event description.

Important constraints:
- You can only use the original query below. You cannot see the video.
- Preserve the original meaning exactly.
- Do not invent specific objects, locations, clothing, colors, counts, camera views, or background details that are not explicitly stated or strongly implied.
- Prefer concrete visual wording when the original query provides it.
- If the original query is generic, keep the rewrite generic instead of hallucinating details.
- Return valid JSON only.

JSON format:
{
  "rewritten_query": "..."
}

Original query:
{{query}}
"""


class GeminiClient:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = os.environ.get(get_config_value(config, "models.gemini.model_env", "GEMINI_MODEL"), "")
        self.api_key = os.environ.get(get_config_value(config, "models.gemini.api_key_env", "GEMINI_API_KEY"), "")
        if not self.model_name or not self.api_key:
            raise RuntimeError("GEMINI_API_KEY and GEMINI_MODEL must be set.")

    def generate_paraphrases(self, query: str) -> List[str]:
        prompt = PARAPHRASE_PROMPT.replace("{{query}}", query)
        max_retries = int(get_config_value(self.config, "models.gemini.max_retries", 2))
        for _ in range(max_retries + 1):
            try:
                payload = _parse_json_object(self._generate_text(prompt))
                paraphrases = _validate_paraphrases(query, payload.get("paraphrases", []))
                if len(paraphrases) == 5:
                    return paraphrases
            except Exception:
                continue
        return _fallback_paraphrases(query)

    def extract_object_constraints(self, query: str) -> List[Dict[str, int]]:
        prompt = OBJECT_PROMPT.replace("{{query}}", query)
        max_retries = int(get_config_value(self.config, "models.gemini.max_retries", 2))
        for _ in range(max_retries + 1):
            try:
                payload = _parse_json_object(self._generate_text(prompt))
                return normalize_coco_constraints(payload.get("objects", []))
            except Exception:
                continue
        return []

    def rewrite_for_keyframe_search(self, query: str) -> str:
        prompt = SCENE_REWRITE_PROMPT.replace("{{query}}", query)
        max_retries = int(get_config_value(self.config, "models.gemini.max_retries", 2))
        for _ in range(max_retries + 1):
            try:
                payload = _parse_json_object(self._generate_text(prompt))
                rewritten = _validate_scene_rewrite(query, payload.get("rewritten_query"))
                if rewritten:
                    return rewritten
            except Exception:
                continue
        return _fallback_scene_rewrite(query)

    def _generate_text(self, prompt: str) -> str:
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            return response.text
        except ImportError:
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError("Install google-generativeai or google-genai for Gemini support.") from exc
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(model=self.model_name, contents=prompt)
            return response.text


def _parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    return json.loads(cleaned)


def _validate_paraphrases(original: str, paraphrases: Any) -> List[str]:
    if not isinstance(paraphrases, list):
        return []
    normalized_original = original.strip().casefold()
    seen = set()
    valid = []
    for item in paraphrases:
        if not isinstance(item, str):
            continue
        text = " ".join(item.split())
        key = text.casefold()
        if not text or key == normalized_original or key in seen:
            continue
        seen.add(key)
        valid.append(text)
    return valid[:5]


def _fallback_paraphrases(query: str) -> List[str]:
    clean = " ".join(query.split()).rstrip(".")
    templates = [
        "Video frame showing {q}.",
        "Keyframe of {q}.",
        "Scene where {q}.",
        "Visual moment showing {q}.",
        "Shot of {q}.",
    ]
    results = []
    seen = set()
    for template in templates:
        text = template.format(q=clean[0].lower() + clean[1:] if clean else clean)
        key = text.casefold()
        if key not in seen and key != query.strip().casefold():
            seen.add(key)
            results.append(text)
    return results[:5]


def _validate_scene_rewrite(original: str, rewritten: Any) -> str:
    if not isinstance(rewritten, str):
        return ""
    text = " ".join(rewritten.split())
    if not text:
        return ""
    if len(text) > 240:
        text = text[:240].rsplit(" ", 1)[0].strip()
    return text


def _fallback_scene_rewrite(query: str) -> str:
    clean = " ".join(query.split()).strip()
    if not clean:
        return clean
    clean = clean.rstrip(".")
    lowered = clean[0].lower() + clean[1:] if clean else clean
    return f"A video keyframe showing {lowered}."


_COCO_SET = set(COCO_CLASSES)
_COCO_ALIASES = {
    "people": "person",
    "man": "person",
    "woman": "person",
    "boy": "person",
    "girl": "person",
    "bike": "bicycle",
    "motorbike": "motorcycle",
    "sofa": "couch",
    "television": "tv",
    "cellphone": "cell phone",
    "phone": "cell phone",
    "dining table": "dining table",
    "table": "dining table",
    "plant": "potted plant",
}


def normalize_coco_constraints(objects: Any) -> List[Dict[str, int]]:
    if not isinstance(objects, list):
        return []
    merged: Dict[str, int] = {}
    for item in objects:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip().lower()
        count = item.get("count")
        if not isinstance(count, int) or count <= 0:
            continue
        name = _normalize_coco_name(name)
        if name in _COCO_SET:
            merged[name] = max(merged.get(name, 0), count)
    return [{"name": name, "count": count} for name, count in sorted(merged.items())]


def _normalize_coco_name(name: str) -> str:
    name = name.replace("_", " ").strip()
    if name in _COCO_ALIASES:
        return _COCO_ALIASES[name]
    if name.endswith("s") and name[:-1] in _COCO_SET:
        return name[:-1]
    return name

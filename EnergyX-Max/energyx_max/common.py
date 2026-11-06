"""Shared settings, file helpers and the local LLM client for the EnergyX-Max pipeline."""

from __future__ import annotations

import json
import re
import time
import requests

from . import config

ROOT = config.path("outputs")  # where the release files go
WORK = config.path("stages")  # each stage's records
PROGRAMS = config.path("programs")
OLLAMA = config.setting("llm", "url")

# One model per agent, each called in its own context. Local Ollama models, set in config/max.yaml.
MODELS = dict(config.setting("llm", "models"))

CATEGORIES = {
    1: "Data Discovery & Availability",
    2: "Data Quality & Preprocessing",
    3: "Consumption & Demand Statistics",
    4: "Temporal & Load-Profile Analytics",
    5: "Building Performance & Normalization",
    6: "Weather & Climate Analytics",
    7: "Spatial & Spatio-Temporal Analytics",
    8: "Forecasting & Predictive Analytics",
    9: "Anomaly & Change Detection",
    10: "Clustering & Similarity Analytics",
}
LEVELS = {
    "L1": "a single lookup or calculation",
    "L2": "basic aggregation, filtering, or ranking",
    "L3": "multiple operations or data sources combined",
    "L4": "analysis across multiple buildings, data sources, or tools",
    "L5": "multi-stage analysis, such as comparing models or methods across a population of buildings",
}
TARGET_PER_CELL = config.setting("pipeline", "target_per_cell")  # 10 categories x 5 levels x 120 = 6,000


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def chat(role: str, system: str, user: str, as_json: bool = False, max_tokens: int = 1500,
         temperature: float | None = None, seed: int | None = None, retries: int = 3) -> str:
    """One call to the role's model, in a fresh context. Returns the reply text."""
    body = {
        "model": MODELS[role],
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "think": False,
        "options": {
            "temperature": config.setting("llm", "temperature") if temperature is None else temperature,
            "seed": config.setting("llm", "seed") if seed is None else seed,
            "num_predict": max_tokens,
            "num_ctx": config.setting("llm", "context_tokens"),
        },
    }
    if as_json:
        body["format"] = "json"
    for attempt in range(retries):
        try:
            reply = requests.post(OLLAMA, json=body, timeout=600)
            reply.raise_for_status()
            return reply.json()["message"]["content"]
        except (requests.RequestException, KeyError):
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def parse_json(text: str) -> dict | None:
    """The first JSON object in a reply, or None."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def extract_code(text: str) -> str:
    """The Python code in a reply: the first fenced block, or the whole reply."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return (match.group(1) if match else text).strip() + "\n"

"""Write the manifest and gold-result models as JSON Schema to schema/.

    uv run python scripts/export_schemas.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from energyx_gold.core.schemas import GoldResult, Task  # noqa: E402

for name, model in (("task", Task), ("gold_result", GoldResult)):
    path = ROOT / "schema" / f"{name}.schema.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(model.model_json_schema(), indent=2) + "\n")
    print(f"wrote {path}")

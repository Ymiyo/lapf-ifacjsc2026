# src/lapf_project/data/text_templates.py

import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "ja"


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_json(name: str) -> list[dict]:
    path = DATA_DIR / name
    return _read_json(path)


def data2dict(data: list[dict]) -> Dict[int, List[str]]:
    d: Dict[int, List[str]] = {}
    for item in data:
        water_level = int(item["water_level"])
        text = item["text"]
        d.setdefault(water_level, []).append(text)
    return d


TEST_DATA = load_json("test_set.json")
TEST_DICT = data2dict(TEST_DATA)

TRAIN_DATA = load_json("train_set.json")
VAL_DATA = load_json("validation_set.json")

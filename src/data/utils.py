from typing import Optional
import os
import json

def try_load_jsonl(path: str) -> Optional[list]:
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return [json.loads(line) for line in fh]

def try_load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)
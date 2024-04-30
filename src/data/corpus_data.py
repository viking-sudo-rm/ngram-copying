import json
import os
from typing import NamedTuple

class CorpusData(NamedTuple):
    prompts_iid: list
    val_iid: list
    prompts_by_domain: list
    val_by_domain: list

    @classmethod
    def load(cls, root):
        with open(os.path.join(root, "data/iid/prompts.jsonl")) as fh:
            prompts_iid = [json.loads(line) for line in fh]
        with open(os.path.join(root, "data/iid/val.jsonl")) as fh:
            val_iid = [json.loads(line) for line in fh]
        with open(os.path.join(root, "data/by-domain/prompts.jsonl")) as fh:
            prompts_by_domain = [json.loads(line) for line in fh]
        with open(os.path.join(root, "data/by-domain/val.jsonl")) as fh:
            val_by_domain = [json.loads(line) for line in fh]
        return cls(prompts_iid, val_iid, prompts_by_domain, val_by_domain)
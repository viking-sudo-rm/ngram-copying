import json
import os
from typing import NamedTuple

class LMGenerations(NamedTuple):
    by_domain: list
    by_model: list
    by_model_p: list
    by_decoding: list

    @classmethod
    def load(cls, root):
        with open(os.path.join(root, "lm-generations/by-domain/pythia-12b.jsonl")) as fh:
            by_domain = [json.loads(line) for line in fh]
        with open(os.path.join(root, "lm-generations/by-model.jsonl")) as fh:
            by_model = [json.loads(line) for line in fh]
        with open(os.path.join(root, "lm-generations/by-model-p.jsonl")) as fh:
            by_model_p = [json.loads(line) for line in fh]
        with open(os.path.join(root, "lm-generations/by-decoding.jsonl")) as fh:
            by_decoding = [json.loads(line) for line in fh]
        return cls(by_domain, by_model, by_model_p, by_decoding)
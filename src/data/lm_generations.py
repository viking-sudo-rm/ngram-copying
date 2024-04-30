import json
import os
from typing import NamedTuple

class LMGenerations(NamedTuple):
    by_domain: list
    by_model: list
    pythia_12b: list

    @classmethod
    def load(cls, root):
        with open(os.path.join(root, "lm-generations/by-domain/pythia-12b.jsonl")) as fh:
            by_domain = [json.loads(line) for line in fh]
        with open(os.path.join(root, "lm-generations/by-model.jsonl")) as fh:
            by_model = [json.loads(line) for line in fh]
        with open(os.path.join(root, "lm-generations/by-model/pythia-12b.jsonl")) as fh:
            pythia_12b = [json.loads(line) for line in fh]
        return cls(by_domain, by_model, pythia_12b)
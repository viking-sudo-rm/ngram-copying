import os
from typing import NamedTuple, Optional

from .utils import try_load_jsonl

class LMGenerations(NamedTuple):
    by_domain: Optional[list]
    by_model: Optional[list]
    by_model_p: Optional[list]
    by_decoding: Optional[list]
    deduped: Optional[list]

    @classmethod
    def load(cls, root):
        return cls(
            by_domain=try_load_jsonl(os.path.join(root, "lm-generations/by-domain/pythia-12b.jsonl")),
            by_model=try_load_jsonl(os.path.join(root, "lm-generations/by-model.jsonl")),
            by_model_p=try_load_jsonl(os.path.join(root, "lm-generations/by-model-p.jsonl")),
            by_decoding=try_load_jsonl(os.path.join(root, "lm-generations/by-decoding.jsonl")),
            deduped=try_load_jsonl(os.path.join(root, "lm-generations/deduped/pythia-12b-deduped.jsonl"))
        )
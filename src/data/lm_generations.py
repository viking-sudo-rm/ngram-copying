import os
from typing import NamedTuple, Optional

from .utils import try_load_jsonl

class LMGenerations(NamedTuple):
    by_domain: Optional[list]
    by_model: Optional[list]
    by_model_p: Optional[list]
    deduped: Optional[list]
    by_topp: Optional[list]
    by_topk: Optional[list]
    by_temp: Optional[list]
    beam1: Optional[list]
    beam4: Optional[list]
    beam8: Optional[list]

    @classmethod
    def load(cls, root):
        results = {
            "by_domain": "by-domain/pythia-12b.jsonl",
            "by_model": "by-model.jsonl",
            "by_model_p": "by-model-p.jsonl",
            "deduped": "deduped/pythia-12b-deduped.jsonl",
            "by_topp": "by-decoding/topp.jsonl",
            "by_topk": "by-decoding/topk.jsonl",
            "by_temp": "by-decoding/temp.jsonl",
            "beam1": "by-decoding/beam1.jsonl",
            "beam4": "by-decoding/beam4.jsonl",
            "beam8": "by-decoding/beam8.jsonl",
        }

        return cls(**{
            name: try_load_jsonl(os.path.join(root, "lm-generations", path))
            for name, path in results.items()
        })
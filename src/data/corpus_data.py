import os
from typing import Optional, NamedTuple

from .utils import try_load_jsonl

class CorpusData(NamedTuple):
    prompts_iid: Optional[list]
    val_iid: Optional[list]
    prompts_by_domain: Optional[list]
    val_by_domain: Optional[list]
    val_reddit: Optional[list]

    @classmethod
    def load(cls, root):
        data = {
            "prompts_iid": "iid/prompts.jsonl",
            "val_iid": "iid/val.jsonl",
            "prompts_by_domain": "by-domain/prompts.jsonl",
            "val_by_domain": "by-domain/val.jsonl",
            "val_reddit": "dolma-reddit/val.jsonl",
        }

        return cls(**{
            name: try_load_jsonl(os.path.join(root, "data", path))
            for name, path in data.items()
        })
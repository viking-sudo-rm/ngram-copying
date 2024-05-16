import os
from typing import Optional, NamedTuple

from .utils import try_load_jsonl

class CorpusData(NamedTuple):
    prompts_iid: Optional[list]
    val_iid: Optional[list]
    prompts_by_domain: Optional[list]
    val_by_domain: Optional[list]

    @classmethod
    def load(cls, root):
        return cls(
            prompts_iid=try_load_jsonl(os.path.join(root, "data/iid/prompts.jsonl")),
            val_iid=try_load_jsonl(os.path.join(root, "data/iid/val.jsonl")),
            prompts_by_domain=try_load_jsonl(os.path.join(root, "data/by-domain/prompts.jsonl")),
            val_by_domain=try_load_jsonl(os.path.join(root, "data/by-domain/val.jsonl")),
        )
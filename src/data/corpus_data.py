import os
from typing import Optional, NamedTuple

from .utils import try_load_jsonl

class CorpusData(NamedTuple):
    prompts_iid: Optional[list]
    val_iid: Optional[list]
    prompts_by_domain: Optional[list]
    val_by_domain: Optional[list]
    val_reddit: Optional[list]
    val_cc: Optional[list]
    val_stack: Optional[list]
    val_pes2o: Optional[list]

    @classmethod
    def load(cls, root):
        data = {
            "prompts_iid": "iid/prompts.jsonl",
            "val_iid": "iid/val.jsonl",
            "prompts_by_domain": "by-domain/prompts.jsonl",
            "val_by_domain": "by-domain/val.jsonl",
            "val_reddit": "dolma/reddit.jsonl",
            "val_cc": "dolma/cc.jsonl",
            "val_stack": "dolma/stack.jsonl",
            "val_pes2o": "dolma/pes2o.jsonl",
        }

        return cls(**{
            name: try_load_jsonl(os.path.join(root, "data", path))
            for name, path in data.items()
        })
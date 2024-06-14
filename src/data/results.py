from typing import Optional
import os
from typing import NamedTuple

from .utils import try_load_json

class Results(NamedTuple):
    val_by_domain: Optional[dict]
    val_iid: Optional[dict]
    val_cl: Optional[dict]
    val_reddit: Optional[dict]
    val_cc: Optional[dict]
    val_stack: Optional[dict]
    val_pes2o: Optional[dict]
    by_domain: Optional[dict]
    by_domain_deduped: Optional[dict]
    by_model: Optional[dict]
    by_model_p: Optional[dict]
    losses: Optional[dict]
    by_topp: Optional[dict]
    by_topk: Optional[dict]
    by_temp: Optional[dict]
    beam1: Optional[dict]
    beam4: Optional[dict]
    beam8: Optional[dict]

    @classmethod
    def load(cls, root):
        results = {
            "val_by_domain": "val.json",
            "val_iid": "val-iid.json",
            "val_cl": "val-cl.json",
            "val_reddit": "dolma/reddit.json",
            "val_cc": "dolma/cc.json",
            "val_stack": "dolma/stack.json",
            "val_pes2o": "dolma/pes2o.json",
            "by_domain": "by-domain.json",
            "by_domain_deduped": "by-domain-deduped.json",
            "by_model": "by-model.json",
            "by_model_p": "by-model-p.json",
            "losses": "perplexity/pythia-12b.json",
            "by_topp": "by-decoding/topp.json",
            "by_topk": "by-decoding/topk.json",
            "by_temp": "by-decoding/temp.json",
            "beam1": "by-decoding/beam1.json",
            "beam4": "by-decoding/beam4.json",
            "beam8": "by-decoding/beam8.json",
        }

        return cls(**{
            name: try_load_json(os.path.join(root, "results", path))
            for name, path in results.items()
        })
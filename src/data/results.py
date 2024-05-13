import json
import os
from typing import NamedTuple

class Results(NamedTuple):
    val_by_domain: dict
    val_iid: dict
    by_domain: dict
    by_domain_deduped: dict
    by_model: dict
    by_model_p: dict
    by_decoding: dict

    @classmethod
    def load(cls, root):
        with open(os.path.join(root, "results/val.json")) as fh:
            val_by_domain = json.load(fh)
        with open(os.path.join(root, "results/val-iid.json")) as fh:
            val_iid = json.load(fh)
        with open(os.path.join(root, "results/by-domain.json")) as fh:
            by_domain = json.load(fh)
        with open(os.path.join(root, "results/by-domain-deduped.json")) as fh:
            by_domain_deduped = json.load(fh)
        with open(os.path.join(root, "results/by-model.json")) as fh:
            by_model = json.load(fh)
        with open(os.path.join(root, "results/by-model-p.json")) as fh:
            by_model_p = json.load(fh)
        with open(os.path.join(root, "results/by-decoding.json")) as fh:
            by_decoding = json.load(fh)
        return cls(val_by_domain, val_iid, by_domain, by_domain_deduped, by_model, by_model_p,
                   by_decoding)
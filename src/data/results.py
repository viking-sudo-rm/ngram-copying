import os
from typing import NamedTuple

from .utils import try_load_json

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
        return cls(
            val_by_domain=try_load_json(os.path.join(root, "results/val.json")),
            val_iid=try_load_json(os.path.join(root, "results/val-iid.json")),
            by_domain=try_load_json(os.path.join(root, "results/by-domain.json")),
            by_domain_deduped=try_load_json(os.path.join(root, "results/by-domain-deduped.json")),
            by_model=try_load_json(os.path.join(root, "results/by-model.json")),
            by_model_p=try_load_json(os.path.join(root, "results/by-model-p.json")),
            by_decoding=try_load_json(os.path.join(root, "results/by-decoding.json")),
        )
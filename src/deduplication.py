import numpy as np

def deduplicate(suffix_contexts, filter_n: int = 30):
    """Remove documents with a large max n-gram"""
    return [d for d in suffix_contexts if max(d) < filter_n]
    # mask = np.ones_like(suffix_contexts)
    # for idx, length in enumerate(suffix_contexts):
    #     if length >= filter_n:
    #         mask[idx + 1 - length: idx + 1] = 0
    # return suffix_contexts[mask.nonzero()]
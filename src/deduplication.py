import numpy as np

def deduplicate(suffix_contexts, filter_n: int = 30):
    """Remove documents with a large max n-gram"""
    deduped = [d for d in suffix_contexts if max(d) < filter_n]
    duped = [d for d in suffix_contexts if max(d) >= filter_n]
    return deduped, duped
    # mask = np.ones_like(suffix_contexts)
    # for idx, length in enumerate(suffix_contexts):
    #     if length >= filter_n:
    #         mask[idx + 1 - length: idx + 1] = 0
    # return suffix_contexts[mask.nonzero()]

def deduplicate_exact(suffix_contexts):
    deduped = [d for d in suffix_contexts if d[-1] < len(d)]
    duped = [d for d in suffix_contexts if d[-1] == len(d)]
    return deduped, duped

def remove_partial_ngrams(suffix_lengths):
    filtered = []
    for lengths in suffix_lengths:
        lengths = np.array(lengths)
        shifted_lengths = np.zeros_like(lengths)
        shifted_lengths[:-1] = lengths[1:]
        nz = np.nonzero(lengths + 1 != shifted_lengths)
        filtered.append(lengths[nz].tolist())
    return filtered
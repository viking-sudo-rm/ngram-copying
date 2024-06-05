"""Methods for computing n-gram novelty data."""

import numpy as np
from collections import Counter

def flatten(lists):
    return [item for row in lists for item in row]

class NgramNovelty:
    def __init__(self, max_n: int = 10):
        self.max_n = max_n

    def get_novelty_lb(self, corpus_size: int, entropy=1.0, prob=1.0):
        lengths = np.arange(self.max_n) + 1
        coeff = np.log2(prob) - entropy
        values = 1. - corpus_size * np.exp2(lengths * coeff)
        return lengths, np.maximum(values, 0)

    def get_proportion_unique(self, suffix_contexts: list[list]):
        """Convert length data to n-gram novelty data"""
        lengths = np.arange(self.max_n)
        counter = Counter(flatten(suffix_contexts))
        freqs = np.array([counter[l] for l in lengths])
        n_novel = np.cumsum(freqs)
        n_total = sum(len(doc) for doc in suffix_contexts)

        # Adjustment: exclude first (n - 1) tokens from beginning of each doc.
        exclude_array = np.array([np.minimum(lengths, len(doc) * np.ones_like(lengths)) for doc in suffix_contexts])
        exclude = np.sum(exclude_array, axis=0)

        # prop_unique = (np.cumsum(freqs) - lengths) / num_ngrams
        prop_unique = (n_novel - exclude) / (n_total - exclude)
        return lengths + 1, prop_unique

"""Methods for computing n-gram novelty data."""

import numpy as np
from collections import Counter

def get_novelty_lb(corpus_size: int, entropy=0.5, prob=1.0, max_n: int = 10):
    lengths = np.arange(max_n) + 1
    coeff = np.log(prob) - entropy
    return lengths, 1. - corpus_size * np.exp(lengths * coeff)

def get_proportion_unique(suffix_contexts, max_n: int = 10):
    """Convert length data to n-gram novelty data"""
    lengths = np.arange(max_n)
    counter = Counter(suffix_contexts)
    freqs = np.array([counter[l] for l in lengths])
    prop_unique = (np.cumsum(freqs) - lengths) / (-lengths + len(suffix_contexts))
    return lengths + 1, prop_unique

# DEPRIORITIZED: do across documents
# def get_confidence_interval(p: float, corpus_size: int, max_n: int = 10):
#     lengths = np.arange(max_n) + 1
#     return np.sqrt(lengths * p * (1 - p) / corpus_size)
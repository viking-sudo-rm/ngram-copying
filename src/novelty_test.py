"""Implements a lower-bound on the n-gram novelty under information-theoretic assumptions."""

import numpy as np

def get_novelty_lb(ns, corpus_size: int, entropy=0.5, prob=1.0):
    coeff = np.log(prob) - entropy
    return 1. - corpus_size * np.exp(ns * coeff)

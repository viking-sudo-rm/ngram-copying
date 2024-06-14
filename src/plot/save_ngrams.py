import os
from collections import defaultdict
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer

from src.data import Results, LMGenerations
from src.data.utils import flatten

PLOTS = "plots"

class SaveNgrams:
    """Save n-grams in model-generated text for qualitative analysis."""
    def __init__(self,
                 ns: list = [8, 16, 32, 64, 128],
                 n_samples: int = 10,
                 seed: int = 42,
                 model: str = "EleutherAI/pythia-12b",
                ):
        self.ns = ns
        self.n_samples = n_samples
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        random.seed(seed)

    def save(self, lmg: LMGenerations, res: Results):
        print("\n===== save n-grams =====")
        plots_dir = os.path.join("tables")
        os.makedirs(plots_dir, exist_ok=True)

        max_n = max(flatten(res.by_model["lengths"]))
        print("max n:", max_n)

        print("Finding n-grams...")
        ngrams = defaultdict(list)
        metas = defaultdict(list)
        ngram_counts = defaultdict(list)
        for idx, (tokens, lengths, counts) in enumerate(zip(tqdm(res.by_model["tokens"]), res.by_model["lengths"], res.by_model["counts"])):
            doc = lmg.by_model[idx]
            meta = doc["meta"]
            if meta["model"] != self.model:
                continue

            lengths = np.array(lengths)
            tokens = np.array(tokens)
            counts = np.array(counts)
            for n in self.ns + [max_n]:
                indices, = (lengths == n).nonzero()
                spans = [tokens[idx + 1 - n: idx + 1] for idx in indices]
                ngrams[n].extend(spans)
                metas[n].extend([meta for _ in spans])
                ngram_counts[n].extend(counts[indices].tolist())

        print("Subsampling n-grams...")
        for n in tqdm(ngrams):
            pairs = list(zip(ngrams[n], metas[n]))
            if len(pairs) > self.n_samples:
                ngrams[n], metas[n] = zip(*random.sample(pairs, self.n_samples))

        print("Saving sampled n-grams...")
        ns = []
        strings = []
        flat_metas = []
        flat_counts = []
        for n in ngrams:
            for span, meta, count in zip(ngrams[n], metas[n], ngram_counts[n]):
                ns.append(n)
                string = self.tokenizer.decode(span)
                strings.append(string)
                flat_metas.append(meta)
                flat_counts.append(count)
        df = pd.DataFrame({
            "n": ns,
            "count": flat_counts,
            "ngram": strings,
            "meta": flat_metas,
        })
        df.to_csv("tables/ngrams.csv")

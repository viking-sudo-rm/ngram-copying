import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from math import log
from collections import defaultdict

from .constants import *
from .utils import *
from ..ngram_novelty import NgramNovelty
from ..data.filter_lengths import FilterLengths
from ..data.utils import flatten

ORANGES = plt.cm.Oranges(np.linspace(0.3, 1, len(PLENS)))

class FrequencyPlots:
    def __init__(self, args, base=10):
        self.args = args
        self.base = base  # Used to determine frequency buckets. Tuned so we get 4

    def plot_frequency(self):
        os.makedirs("plots/frequency", exist_ok=True)
        counts_12b = FilterLengths(lmg, results).get_lengths_for_model("EleutherAI/pythia-12b", key="counts")
        plt.figure()
        max_n = self.args.max_n

        # Plot the validation set as a baseline here.
        df = pd.DataFrame({
            "lengths": flatten(lengths_12b["val"]),
            "counts": flatten(counts_12b["val"]),
        })
        stats = df.groupby("lengths").median()
        plt.plot(stats.index[:max_n], stats["counts"][:max_n], label="val", linestyle="--", color=GRAY)

        # Find the threshold at which copying happens.
        sizes, prop_unique = nov.get_novelty_lb(CORPUS_SIZE, entropy=ENTROPY, prob=P_AMBIGUOUS)
        idx = np.argmin(np.abs(prop_unique - 0.5))
        threshold = sizes[idx]
        plt.axvline(threshold, linestyle="--", color=MUTED_RED, label="LB")

        for plen, color in zip(PLENS, ORANGES):
            df = pd.DataFrame({
                "lengths": flatten(lengths_12b[plen]),
                "counts": flatten(counts_12b[plen]),
            })
            # results = df.groupby("lengths").describe()
            stats = df.groupby("lengths").median()
            plt.plot(stats.index[:max_n], stats["counts"][:max_n], label=f"p={plen}", color=color)
        plt.xlabel("max suffix length")
        plt.ylabel("median suffix frequency")
        plt.yscale("log")
        if args.log_scale:
            plt.xscale("log")
        plt.tight_layout()
        plt.legend()
        plt.savefig("plots/frequency/freq.pdf")

    def plot_novelty_by_freq(self, lengths_12b, lmg, results):
        os.makedirs("plots/frequency", exist_ok=True)
        nov = NgramNovelty(self.args.max_n)
        counts_12b = FilterLengths(lmg, results).get_lengths_for_model("EleutherAI/pythia-12b", key="counts")

        # Filter documents by median count in the document.
        lengths_by_oom = defaultdict(list)
        for lengths, counts in zip(lengths_12b[0], counts_12b[0]):
            med_count = np.median(counts)
            oom = round(log(med_count) / log(self.base))
            lengths_by_oom[oom].append(lengths)

        ooms = list(sorted(lengths_by_oom.keys()))
        colors = plt.cm.Reds(np.linspace(0.3, 1, len(lengths_by_oom)))

        plt.figure()
        plot_baselines(plt, nov, lengths_12b["val"], lengths_12b["val_reddit"])
        for color, oom in zip(colors, ooms):
            lengths = lengths_by_oom[oom]
            sizes, prop_unique = nov.get_proportion_unique(lengths)
            plt.plot(sizes, prop_unique, color=color, label="$" + str(self.base) + "^{" + str(oom) + "}$")
        format_novelty_plot(self.args, plt)
        plt.legend(title="Frequency")
        plt.savefig("plots/frequency/novelty.pdf")
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

from ..data.filter_lengths import FilterLengths
from .utils import plot_baselines, format_novelty_plot, clean_model_name
from .constants import *
from ..ngram_novelty import NgramNovelty, flatten
from ..deduplication import remove_partial_ngrams

ORANGES = plt.cm.Oranges(np.linspace(0.3, 1, len(PLENS)))

def get_nnsl_by_index(lengths):
        max_len = max(len(doc) for doc in lengths)
        lengths_by_idx = [[] for _ in range(max_len)]
        for doc in lengths:
            for idx, length in enumerate(doc):
                lengths_by_idx[idx].append(length)
        return lengths_by_idx

class MainPlots:
    def __init__(self, args, model="EleutherAI/pythia-12b", name="Pythia-12B"):
        self.args = args
        self.model = model
        self.name = name

    def plot_model(self, lengths_12b, lmg, results):
        os.makedirs("plots/main", exist_ok=True)
        nov = NgramNovelty(self.args.max_n)

        print("\n\n=== plot_model ===")
        if self.model == "EleutherAI/pythia-12b":
            lengths = lengths_12b
        else:
            lengths = FilterLengths(lmg, results).get_lengths_for_model(self.model)
        
        plt.figure()
        lengths_dolma = lengths_12b["val_reddit"] + lengths_12b["val_pes2o"]
        _, red_unique, _ = plot_baselines(plt, nov, lengths["val"], lengths_dolma, show_lb=True)
        sizes, prop_unique = nov.get_proportion_unique(lengths[0])
        plt.plot(sizes, prop_unique, color=DARK_GREEN, label="Pythia-12B")

        for n in [3, 10, 100]:
            print(f"  {n}: pythia={prop_unique[(sizes == n).nonzero()].item():.2f}, dolma={red_unique[(sizes == n).nonzero()].item():.2f}")

        format_novelty_plot(self.args, plt)

        print("First idx less novel than Reddit:", np.nonzero(prop_unique < red_unique)[0][0])
        plt.savefig(f"plots/main/{clean_model_name(self.model)}.pdf")
        plt.close()

        # Histogram plots.
        plt.figure()
        kwargs = dict(bins=100, alpha=.5, density=True)

        for split in ["val", "val_reddit", "val_cc", "val_stack", "val_pes2o"]:
            data = np.array(flatten(remove_partial_ngrams(lengths[split])))
            print(f"{split}: med={np.median(data)}, mean={np.mean(data):.2f}, max={np.max(data)}")
            plt.hist(np.log(data), label=split, **kwargs)

        gdata = np.array(flatten(remove_partial_ngrams(lengths[0])))
        gdata = np.array([x for x in gdata if x != 0])
        plt.hist(np.log(gdata), label=clean_model_name(self.model), **kwargs)

        plt.xlabel("non-novel suffix length")
        plt.legend()
        plt.tight_layout()
        ticks = [1, 10, 100]
        plt.xticks(np.log(ticks), ticks)
        # plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.savefig(f"plots/main/histogram.pdf")
        plt.close()
    
    def plot_by_seq_len(self, lengths_12b):
        os.makedirs("plots/main", exist_ok=True)

        indices = np.arange(1000) + 1
        data = {
            "pythia": get_nnsl_by_index(lengths_12b[0]),
            "val": get_nnsl_by_index(lengths_12b["val"]),
            # "dolma": get_nnsl_by_index(lengths_12b["dolma"]),
        } 
        colors = ["green", "red"]

        plt.figure()
        for color, (name, lengths) in zip(colors, data.items()):
            med_lengths = [np.median(li) for li in lengths]
            mean_lengths = [np.mean(li) for li in lengths]
            plt.plot(indices, med_lengths, color=color, label=name + " (med)", linestyle="--")
            plt.plot(indices, mean_lengths, color=color, label=name + " (mean)", linestyle="-")
        plt.xscale("log")
        plt.xlabel("Token index")
        plt.ylabel("Non-novel suffix length")
        plt.tight_layout()
        plt.legend()
        sns.despine()
        plt.savefig(f"plots/main/nnsl-by-index.pdf")

        plt.figure()
        ns = [2, 8, 32, 128, 512]
        colors = plt.cm.Oranges(np.linspace(0.3, 1.0, len(ns)))
        for n, color in zip(ns, colors):
            p_uniques = [(np.array(lengths) > n).nonzero()[0].shape[0] / len(lengths) for lengths in data["val"]]
            plt.plot(indices[n:], p_uniques[n:], color=color, label=f"{n}-grams")
        plt.xscale("log")
        plt.xlabel("Token index")
        plt.ylabel("% novel")
        plt.tight_layout()
        plt.legend()
        sns.despine()
        plt.savefig(f"plots/main/novelty-by-index.pdf")
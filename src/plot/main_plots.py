import matplotlib.pyplot as plt
import numpy as np
import os

from ..data.filter_lengths import FilterLengths
from .utils import plot_baselines, format_novelty_plot, clean_model_name
from .constants import *
from ..ngram_novelty import NgramNovelty, flatten
from ..deduplication import remove_partial_ngrams

ORANGES = plt.cm.Oranges(np.linspace(0.3, 1, len(PLENS)))

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
        
        # DEDUP STUFF
        # deduped, duped = deduplicate_exact(val)
        # sizes, prop_unique = nov.get_proportion_unique(deduped)
        # plt.plot(sizes, prop_unique, linestyle="--", color="purple", label=f"exact")
        # print("# exact duped:", len(duped), "/", len(val))
        # FILTERS = [float("inf"), 100, 50, 30, 25]
        # colors = plt.cm.Greys(np.linspace(0.2, 1.0, len(FILTERS)))
        # for color, filter_n in zip(colors, FILTERS):
        #     deduped, duped = deduplicate(val, filter_n)
        #     print(f"# duped @ {filter_n}:", len(duped), "/", len(val))
        #     sizes, prop_unique = nov.get_proportion_unique(deduped)
        #     plt.plot(sizes, prop_unique, linestyle="--", color=color, label=f"val@{filter_n}")
        
        plt.figure()
        _, red_unique, _ = plot_baselines(plt, nov, lengths["val"], lengths["val_reddit"])
        # for color, plen in zip(ORANGES, PLENS):
        #     sizes, prop_unique = nov.get_proportion_unique(lengths[plen])
        #     plt.plot(sizes, prop_unique, color=color, label=f"p={plen}")
        sizes, prop_unique = nov.get_proportion_unique(lengths[0])
        plt.plot(sizes, prop_unique, color=DARK_GREEN, label="Pythia-12B")
        format_novelty_plot(self.args, plt)

        print("First idx less novel than Reddit:", np.nonzero(prop_unique < red_unique)[0][0])
        plt.savefig(f"plots/main/{clean_model_name(self.model)}.pdf")
        plt.close()

        # Histogram plots.
        vdata = np.array(flatten(remove_partial_ngrams(lengths["val"])))
        rdata = np.array(flatten(remove_partial_ngrams(lengths["val_reddit"])))
        gdata = np.array(flatten(remove_partial_ngrams(lengths[0])))
        gdata = np.array([x for x in gdata if x != 0])
        print("med(gen):", np.median(gdata))
        print("med(red):", np.median(gdata))
        print("med(val):", np.mean(vdata))

        plt.figure()
        kwargs = dict(bins=100, alpha=.5, density=True)
        plt.hist(np.log(vdata), label="val", **kwargs)
        plt.hist(np.log(rdata), label="reddit", **kwargs)
        plt.hist(np.log(gdata), label=clean_model_name(self.model), **kwargs)
        plt.xlabel("non-novel suffix length")
        plt.legend()
        plt.tight_layout()
        ticks = [1, 10, 100]
        plt.xticks(np.log(ticks), ticks)
        # plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.savefig(f"plots/main/histogram.pdf")
        plt.close()
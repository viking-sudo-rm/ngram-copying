import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import floor, log
from scipy.signal import savgol_filter
from copy import deepcopy
import seaborn as sns

from . import *
from ..data import Results
from ..data.utils import try_load_json

PLOTS = "plots"
RESULTS = "/net/nfs.cirrascale/allennlp/willm/ngram-copying/results"

MODELS = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b", "pythia-2.8b",
          "pythia-6.9b", "pythia-12b"]

BLUES = plt.cm.Blues(np.linspace(0.2, 1, len(MODELS)))
REDS = plt.cm.Reds(np.linspace(0.2, 1, len(MODELS)))

FILTER_SIZE = 100
FILTER_ORDER = 2

def smooth(data):
    return data
    # return savgol_filter(data, FILTER_SIZE, FILTER_ORDER)

def flatten(lists):
    return [item for row in lists for item in row]

class CompletionLossPlots:
    def __init__(self, args):
        self.args = args

    def get_completed_lengths_stats(self, lengths, counts, big=False):
        # The length and count of the target token completing the n-gram.
        lengths = flatten([li[1:] for li in lengths])
        counts = flatten([li[1:] for li in counts])

        # Compute losses.
        dirname = os.path.join(RESULTS, "completion-loss")
        all_stats = {}
        for file in os.listdir(dirname):
            if (big and "BIG" not in file) or (not big and "BIG" in file):
                continue
            res = try_load_json(os.path.join(dirname, file))
            model = res["model"]
            losses = flatten(res["losses"])
            df = pd.DataFrame({
                "lengths": lengths,
                "counts": counts,
                "losses": losses,
            })
            stats = df.groupby("lengths").mean()
            all_stats[model.split("/")[1]] = stats
        
        return all_stats

    def get_uncompleted_lengths_stats(self, lengths, counts, big=False):
        indices = []
        uncompleted_lengths = []
        uncompleted_counts = []

        for doc_lengths, doc_counts in zip(lengths, counts):
            doc_lengths = np.array(doc_lengths)
            doc_counts = np.array(doc_counts)
            nz, = np.nonzero(doc_lengths[:-1] + 1 != doc_lengths[1:])
            uncompleted_lengths.extend(doc_lengths[nz])
            uncompleted_counts.extend(doc_counts[nz + 1])
            indices.append(nz)

        # For comparability, we include the token in the n-gram size.
        uncompleted_lengths = np.array(uncompleted_lengths) + 1

        # Compute losses.
        dirname = os.path.join(RESULTS, "completion-loss")
        all_stats = {}
        for file in os.listdir(dirname):
            if (big and "BIG" not in file) or (not big and "BIG" in file):
                continue
            res = try_load_json(os.path.join(dirname, file))
            model = res["model"]
            losses = flatten([np.array(li)[nz] for li, nz in zip(res["losses"], indices)])
            df = pd.DataFrame({
                "lengths": uncompleted_lengths,
                "counts": uncompleted_counts,
                "losses": losses,
            })
            stats = df.groupby("lengths").mean()
            all_stats[model.split("/")[1]] = stats
        
        return all_stats

    def plot_by_model(self, results: Results):
        plots_dir = os.path.join(PLOTS, "completion-loss")
        os.makedirs(plots_dir, exist_ok=True)
        args = deepcopy(self.args)
        args.max_n = 10
        args.log_scale = False

        # Compute stats from CDAWG output.
        lengths = results.val_iid["lengths"]
        counts = results.val_iid["counts"]
        completed = self.get_completed_lengths_stats(lengths, counts)
        uncompleted = self.get_uncompleted_lengths_stats(lengths, counts)

        plt.figure()
        max_n = args.max_n
        for color, model in zip(BLUES, MODELS):
            stats = completed[model]
            plt.plot(stats.index[:max_n], smooth(stats["losses"][:max_n]), color=color, label=model)
        for color, model in zip(REDS, MODELS):
            stats = uncompleted[model]
            # Substract 1 from max-n because it starts one later.
            plt.plot(stats.index[:max_n - 1], smooth(stats["losses"][:max_n - 1]), color=color)
        plt.xlabel("non-novel suffix length")
        plt.ylabel("mean completion loss")
        if args.log_scale:
            plt.xscale("log")
            max_e = floor(log(max_n + 1, 10))
            ticks = [10**e for e in range(0, max_e + 1)]
            plt.xticks(ticks)
            plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        else:
            plt.xticks(list(range(1, max_n + 1)))
        plt.tight_layout()
        # plt.legend()
        plt.savefig(os.path.join(plots_dir, "by-model.pdf"))

        # fig, ax1 = plt.subplots(1, 1)
        # ax1.plot(stats.index[:args.max_n], stats["losses"][:args.max_n], color="orange")
        # ax1.set_ylabel("mean token loss", color="orange")
        # # ax1.tick_params("y", color="orange")

        # ax2 = plt.twinx()
        # ax2.plot(stats.index[:args.max_n], stats["counts"][:args.max_n], color=GRAY, linestyle="--")
        # ax2.set_yscale("log")
        # ax2.set_ylabel("mean pretrain. freq.", color=GRAY)
        # # ax2.tick_params("y", colors=GRAY)

        # plt.xlabel("max suffix length")
        # # plt.ylabel(f"mean token loss ({model})")
        # if args.log_scale:
        #     plt.xscale("log")
        # plt.tight_layout()
        # plt.savefig("plots/loss/loss.pdf")

    def plot_big(self, results: Results):
        plots_dir = os.path.join(PLOTS, "completion-loss")
        os.makedirs(plots_dir, exist_ok=True)

        # Compute stats from CDAWG output.
        lengths = results.val_cl["lengths"]
        counts = results.val_cl["counts"]
        completed = self.get_completed_lengths_stats(lengths, counts, big=True)
        uncompleted = self.get_uncompleted_lengths_stats(lengths, counts, big=True)

        plt.figure()
        max_n = self.args.max_n

        stats = completed["pythia-12b"]
        plt.plot(stats.index[:max_n], smooth(stats["losses"][:max_n]), color="blue", label="in train")

        stats = uncompleted["pythia-12b"]
        # Substract 1 from max-n because it starts one later.
        plt.plot(stats.index[:max_n - 1], smooth(stats["losses"][:max_n - 1]), color="red", label="not in train")

        plt.xlabel("$n$-gram size")
        plt.ylabel("completion loss")
        if self.args.log_scale:
            plt.xscale("log")
            max_e = floor(log(max_n + 1, 10))
            ticks = [10**e for e in range(0, max_e + 1)]
            plt.xticks(ticks)
            plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        else:
            plt.xticks(list(range(1, max_n + 1)))
        sns.despine()
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(plots_dir, "big.pdf"))

    def plot_by_freq(self, results: Results):
        plots_dir = os.path.join(PLOTS, "completion-loss")
        os.makedirs(plots_dir, exist_ok=True)

        lengths = results.val_cl["lengths"]
        counts = results.val_cl["counts"]
        losses_path = os.path.join(RESULTS, "completion-loss", "pythia-12b-BIG.json")
        losses = try_load_json(losses_path)["losses"]

        lengths = np.array(flatten([li[1:] for li in lengths]))
        counts = np.array(flatten([li[1:] for li in counts]))
        losses = np.array(flatten(losses))

        ns = [4, 8, 16, 32, 64, 128]
        oranges = plt.cm.Oranges(np.linspace(0.2, 1, len(ns)))

        plt.figure()
        for color, n in zip(oranges, ns):
            mask = (lengths == n)
            mcounts = counts[mask.nonzero()]
            mlosses = losses[mask.nonzero()]
            oom = np.round(np.log10(mcounts))
            df = pd.DataFrame({
                "freq-oom": oom,
                "loss": mlosses,
            })
            stats = df.groupby("freq-oom").mean()
            plt.plot([10**x for x in stats.index], stats["loss"], color=color, marker=".", label=f"{n}-grams")
        plt.xlabel("$n$-gram frequency")
        plt.ylabel("mean completion loss")
        plt.xscale("log")
        plt.legend()
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS, "completion-loss", f"loss-vs-freq.pdf"))

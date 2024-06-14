"""Script to regenerate all of the plots that we need."""

from collections import defaultdict
from typing import NamedTuple
import numpy as np
import argparse
import json
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from copy import deepcopy
import matplotlib.ticker
from matplotlib.patches import Rectangle

from src.plot import *
from src.plot.by_decoding import DecodingPlots
from src.plot.completion_loss import CompletionLossPlots
from src.plot.main_plots import MainPlots
from src.plot.example import ExamplePlots
from src.plot.save_table import SaveTable
from src.plot.save_ngrams import SaveNgrams
from src.plot.frequency import FrequencyPlots
from src.data import CorpusData, LMGenerations, Results
from src.data.filter_lengths import FilterLengths
from src.ngram_novelty import NgramNovelty, flatten

plt.rcParams.update({'font.size': 13})

MODELS = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b", "pythia-2.8b",
          "pythia-6.9b", "pythia-12b"]

BLUES = plt.cm.Blues(np.linspace(0.2, 1, len(MODELS)))
ORANGES = plt.cm.Oranges(np.linspace(0.3, 1, len(PLENS)))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str,
                        default="/net/nfs.cirrascale/allennlp/willm/ngram-copying")
    parser.add_argument("--max-n", "-n", type=int, default=10)
    parser.add_argument("--log-scale", action="store_true")
    parser.add_argument("--no-plen", action="store_true", help="Hide plen from titles")
    return parser.parse_args()

def plot_by_plen(args, model="EleutherAI/pythia-12b"):
    if model == "EleutherAI/pythia-12b":
        lengths = lengths_12b
    else:
        lengths = FilterLengths(lmg, results).get_lengths_for_model(model)

    os.makedirs("plots/by-plen", exist_ok=True)

    plt.figure()
    # Should we just average????
    buckets = [[np.mean(doc) for doc in lengths[plen]] for plen in ["val"] + PLENS]
    plt.violinplot(buckets, showmeans=True)
    plt.xticks(range(1, len(PLENS) + 2), ["val"] + PLENS)
    plt.xlabel("prompt length")
    plt.ylabel("Mean NNSL")
    plt.yscale("log")
    plt.savefig("plots/by-plen/violin.pdf")
    plt.close()

    plt.figure()
    mean_lengths = [np.mean(flatten(lengths[plen])) for plen in PLENS]
    plt.scatter(PLENS, mean_lengths)
    plt.xticks(range(1, len(PLENS) + 2), ["val"] + PLENS)
    plt.xlabel("prompt length")
    plt.ylabel("mean NNSL")
    plt.xscale("symlog")
    plt.savefig("plots/by-plen/scatter.pdf")
    plt.close()

def plot_by_model(args, filter_fn=None):
    os.makedirs("plots/by-model", exist_ok=True)
    
    # TODO: Use FilterLengths
    plot_lengths = defaultdict(list)
    plot_lengths["val"] = results.val_iid["lengths"]
    plot_lengths["val_reddit"] = results.val_reddit["lengths"]
    for doc, lengths in zip(lmg.by_model, results.by_model["lengths"]):
        if filter_fn and not filter_fn(doc["meta"]):
            continue
        model = clean_model_name(doc["meta"]["model"])
        plot_lengths[model].append(lengths)

    plt.figure()
    lengths_dolma = lengths_12b["val_reddit"] + lengths_12b["val_pes2o"]
    plot_baselines(plt, nov, lengths_12b["val"], lengths_dolma)
    for color, key in zip(BLUES, MODELS):
        sizes, prop_unique = nov.get_proportion_unique(plot_lengths[key])
        plt.plot(sizes, prop_unique, label=clean_size_string(key), color=color)
    format_novelty_plot(args, plt)
    plt.savefig("plots/by-model/curves.pdf")
    plt.close()
    
    sizes = []
    mean_lengths = []
    for model, lengths in plot_lengths.items():
        if "val" in model:
            continue
        sizes.append(clean_size(model.split("-")[-1]))
        mean_lengths.append(np.mean(flatten(lengths)))

    pairs = list(zip(sizes, mean_lengths))
    pairs.sort()
    sizes, mean_lengths = zip(*pairs)

    mean_red = np.mean(flatten(plot_lengths["val_reddit"]))

    plt.figure()
    plt.axhline(mean_red, color=GRAY, linestyle="--")
    for color, size, length in zip(BLUES, sizes, mean_lengths):
        plt.scatter(size, length, s=100, marker=".", color=color)
    # plt.plot(sizes, mean_lengths, linestyle="-", marker=".", color=BLUES[-1])
    plt.xlabel("Model size")
    plt.ylabel("Mean NNSL")
    plt.xscale("log")
    plt.tight_layout()
    sns.despine()
    plt.savefig("plots/by-model/scatter.pdf")
    plt.close()

def plot_by_domain(args, deduped=False):
    dirname = "plots/by-domain" + ("-deduped" if deduped else "")
    os.makedirs(dirname, exist_ok=True)
    lengths_by_domain = defaultdict(list)
    domains = set()

    # First compute for validation set.
    for doc, lengths in zip(data.val_by_domain, results.val_by_domain["lengths"]):
        domain = doc["meta"]["pile_set_name"]
        lengths_by_domain[domain, "val"].append(lengths)
        domains.add(domain)

    # Next compute for CDAWG outputs.
    all_lengths = results.by_domain["lengths"] if not deduped else results.by_domain_deduped["lengths"]
    for doc, lengths in zip(lmg.by_domain, all_lengths):
        plen = doc["meta"]["prompt_len"]
        pidx = doc["meta"]["prompt_idx"]
        prompt = data.prompts_by_domain[pidx]
        domain = prompt["meta"]["pile_set_name"]
        lengths_by_domain[domain, plen].append(lengths)
        domains.add(domain)

    for domain in domains:
        plt.figure()

        # Validation set baseline.
        val_lengths = lengths_by_domain[domain, "val"]
        val_sizes, val_prop_unique = nov.get_proportion_unique(val_lengths)
        plt.plot(val_sizes, val_prop_unique, linestyle="--", color=GRAY, label="val")

        # Union bound baseline.
        sizes, prop_unique = nov.get_novelty_lb(CORPUS_SIZE, entropy=ENTROPY, prob=P_AMBIGUOUS)
        plt.plot(sizes, prop_unique, linestyle="--", color=MUTED_RED, label="LB")

        for color, plen in zip(ORANGES, PLENS):
            if plen == 0:
                lengths = lengths_12b[0]
            else:
                lengths = lengths_by_domain[domain, plen]
            sizes, prop_unique = nov.get_proportion_unique(lengths)
            plt.plot(sizes, prop_unique, label=f"p={plen}", color=color)
        plt.title(f"n-gram novelty for {domain}" + (" (deduped)" if deduped else ""))
        format_novelty_plot(args, plt)
        plt.savefig(f"{dirname}/{domain}.pdf")
        plt.close()

if __name__ == "__main__":
    args = parse_args()
    nov = NgramNovelty(args.max_n)
    sizes, prop_unique = nov.get_novelty_lb(CORPUS_SIZE, entropy=ENTROPY, prob=P_AMBIGUOUS)
    idx = np.argmin(np.abs(prop_unique - 0.5))
    threshold = sizes[idx]
    print("LB threshold:", threshold)

    # Data used by all plots. 
    data = CorpusData.load(args.root)
    lmg = LMGenerations.load(args.root)
    results = Results.load(args.root)
    lengths_12b = FilterLengths(lmg, results).get_lengths_for_model("EleutherAI/pythia-12b")

    # Example plots.
    example = ExamplePlots(args)
    example.plot_example()

    # Main plots.
    main = MainPlots(args)
    main.plot_model(lengths_12b, lmg, results)
    main.plot_by_seq_len(lengths_12b)
    plot_by_plen(args)

    # Frequency plots.
    # freq = FrequencyPlots(args, base=10)
    # # freq.plot_frequency()
    # freq.plot_novelty_by_freq(lengths_12b, lmg, results)

    # Model size.
    plot_by_model(args)

    # # Plot by domain.
    # plot_by_domain(args)
    # plot_by_domain(args, deduped=True)

    # # Decoding experiments.
    decoding = DecodingPlots(args)
    decoding.plot_by_topk(lmg, results, lengths_12b)
    decoding.plot_by_topp(lmg, results, lengths_12b)
    decoding.plot_by_temp(lmg, results, lengths_12b)
    decoding.plot_by_beam(lmg, results, lengths_12b)

    # Completion loss experiments.
    closs = CompletionLossPlots(args)
    # closs.plot_by_model(results)
    closs.plot_big(results)
    closs.plot_by_freq(results)

    # Table of mean NNSL results.
    table = SaveTable()
    table.save(lmg, results)

    # Example n-grams.
    # ngrams = SaveNgrams(n_samples=50)
    # ngrams.save(lmg, results)
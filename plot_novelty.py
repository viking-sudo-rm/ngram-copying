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

from src.plots import *
from src.plots.by_decoding import DecodingPlots
from src.plots.completion_loss import CompletionLossPlots
from src.data import CorpusData, LMGenerations, Results
from src.ngram_novelty import NgramNovelty
from src.deduplication import deduplicate

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

def get_lengths_for_model(model: str, key="lengths") -> dict:
    plot_lengths = defaultdict(list)
    plot_lengths["val"] = results.val_iid[key]
    for doc, lengths in zip(lmg.by_model, results.by_model[key]):
        if doc["meta"]["model"] != model:
            continue
        plen = doc["meta"]["prompt_len"]
        plot_lengths[plen].append(lengths)
    for doc, lengths in zip(lmg.by_model_p, results.by_model_p[key]):
        if doc["meta"]["model"] != model:
            continue
        plen = doc["meta"]["prompt_len"]
        plot_lengths[plen].append(lengths)
    return plot_lengths

def plot_model(args, model="EleutherAI/pythia-12b", name="Pythia-12B"):
    if model == "EleutherAI/pythia-12b":
        lengths = lengths_12b
    else:
        lengths = get_lengths_for_model(model)

    plt.figure()
    
    # Validation baseline.
    val = lengths["val"]
    # sizes, prop_unique = nov.get_proportion_unique(val)
    # plt.plot(sizes, prop_unique, linestyle="--", color=GRAY, label="val")
    
    FILTERS = [float("inf"), 1000, 100, 50, 40, 30, 25, 20, 15, 10]
    colors = plt.cm.Greys(np.linspace(0.2, 1.0, len(FILTERS)))
    for color, filter_n in zip(colors, FILTERS):
        deduped = deduplicate(val, filter_n)
        sizes, prop_unique = nov.get_proportion_unique(deduped)
        plt.plot(sizes, prop_unique, linestyle="--", color=color, label=f"val@{filter_n}")

        if isinstance(filter_n, int):
            # FIXME: why isn't this 100%?
            print("===", filter_n, "===")
            print(sizes[:filter_n])
            print(prop_unique[:filter_n])
            print("Max before filter:", np.max(prop_unique[:filter_n - 1]))

    # Union bound baseline.
    sizes, prop_unique = nov.get_novelty_lb(CORPUS_SIZE, entropy=ENTROPY, prob=P_AMBIGUOUS)
    plt.plot(sizes, prop_unique, linestyle="--", color=MUTED_RED, label="LB")
    
    for color, plen in zip(ORANGES, PLENS):
        sizes, prop_unique = nov.get_proportion_unique(lengths[plen])
        plt.plot(sizes, prop_unique, color=color, label=f"p={plen}")
    format_novelty_plot(args, plt)
    plt.title(f"n-gram novelty of {name}")

    os.makedirs("plots/by-model", exist_ok=True)
    plt.savefig(f"plots/by-model/{clean_model_name(model)}.pdf")
    plt.close()

def plot_by_plen(args, model="EleutherAI/pythia-12b"):
    if model == "EleutherAI/pythia-12b":
        lengths = lengths_12b
    else:
        lengths = get_lengths_for_model(model)

    os.makedirs("plots/by-plen", exist_ok=True)

    plt.figure()
    # Should we just average????
    buckets = [[np.mean(doc) for doc in lengths[plen]] for plen in ["val"] + PLENS]
    plt.violinplot(buckets, showmeans=True)
    plt.xticks(range(1, len(PLENS) + 2), ["val"] + PLENS)
    plt.xlabel("prompt length")
    plt.ylabel("mean non-novel suffix length")
    plt.yscale("log")
    plt.savefig("plots/by-plen/violin.pdf")
    plt.close()

    plt.figure()
    mean_lengths = [np.mean(flatten(lengths[plen])) for plen in PLENS]
    plt.scatter(PLENS, mean_lengths)
    plt.xticks(range(1, len(PLENS) + 2), ["val"] + PLENS)
    plt.xlabel("prompt length")
    plt.ylabel("mean non-novel suffix length")
    plt.xscale("symlog")
    plt.savefig("plots/by-plen/scatter.pdf")
    plt.close()

def plot_by_model(args, filter_fn=None):
    os.makedirs("plots/by-model", exist_ok=True)
    
    plot_lengths = defaultdict(list)
    plot_lengths["val"] = results.val_iid["lengths"]
    for doc, lengths in zip(lmg.by_model, results.by_model["lengths"]):
        if filter_fn and not filter_fn(doc["meta"]):
            continue
        model = clean_model_name(doc["meta"]["model"])
        plot_lengths[model].append(lengths)

    # TODO: Make this into a violinplot at some point?
    plt.figure()
    sizes, prop_unique = get_proportion_unique(flatten(plot_lengths["val"]), max_n=args.max_n)
    plt.plot(sizes, prop_unique, label="val", color=GRAY, linestyle="--")
    sizes, prop_unique = get_novelty_lb(CORPUS_SIZE, entropy=ENTROPY, prob=P_AMBIGUOUS, max_n=args.max_n)
    plt.plot(sizes, prop_unique, linestyle="--", color=MUTED_RED, label="LB")
    for color, key in zip(BLUES, MODELS):
        sizes, prop_unique = get_proportion_unique(flatten(plot_lengths[key]), max_n=args.max_n)
        plt.plot(sizes, prop_unique, label=key.split("-")[-1], color=color)
    format_novelty_plot(args, plt)
    plt.savefig("plots/by-model/curves.pdf")
    plt.close()
    
    sizes = []
    mean_lengths = []
    for model, lengths in plot_lengths.items():
        if model == "val":
            continue
        sizes.append(clean_size(model.split("-")[-1]))
        mean_lengths.append(np.mean(flatten(lengths)))

    plt.figure()
    plt.scatter(sizes, mean_lengths, linestyle="-")
    plt.xlabel("model size")
    plt.ylabel("mean non-novel suffix length")
    plt.xscale("log")
    plt.tight_layout()
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
        val_sizes, val_prop_unique = get_proportion_unique(flatten(val_lengths), max_n=args.max_n)
        plt.plot(val_sizes, val_prop_unique, linestyle="--", color=GRAY, label="val")

        # Union bound baseline.
        sizes, prop_unique = get_novelty_lb(CORPUS_SIZE, entropy=ENTROPY, prob=P_AMBIGUOUS, max_n=args.max_n)
        plt.plot(sizes, prop_unique, linestyle="--", color=MUTED_RED, label="LB")

        for color, plen in zip(ORANGES, PLENS):
            if plen == 0:
                lengths = lengths_12b[0]
            else:
                lengths = lengths_by_domain[domain, plen]
            sizes, prop_unique = get_proportion_unique(flatten(lengths), max_n=args.max_n)
            plt.plot(sizes, prop_unique, label=f"p={plen}", color=color)
        plt.title(f"n-gram novelty for {domain}" + (" (deduped)" if deduped else ""))
        format_novelty_plot(args, plt)
        plt.savefig(f"{dirname}/{domain}.pdf")
        plt.close()

def plot_frequency(args):
    os.makedirs("plots/frequency", exist_ok=True)
    counts_12b = get_lengths_for_model("EleutherAI/pythia-12b", key="counts")
    plt.figure()

    # Plot the validation set as a baseline here.
    df = pd.DataFrame({
        "lengths": flatten(lengths_12b["val"]),
        "counts": flatten(counts_12b["val"]),
    })
    stats = df.groupby("lengths").median()
    plt.plot(stats.index[:args.max_n], stats["counts"][:args.max_n], label="val", linestyle="--", color=GRAY)

    # Find the threshold at which copying happens.
    sizes, prop_unique = get_novelty_lb(CORPUS_SIZE, entropy=ENTROPY, prob=P_AMBIGUOUS, max_n=args.max_n)
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
        plt.plot(stats.index[:args.max_n], stats["counts"][:args.max_n], label=f"p={plen}", color=color)
    plt.xlabel("max suffix length")
    plt.ylabel("median suffix frequency")
    plt.yscale("log")
    if args.log_scale:
        plt.xscale("log")
    plt.tight_layout()
    plt.legend()
    plt.savefig("plots/frequency/freq.pdf")

if __name__ == "__main__":
    args = parse_args()
    nov = NgramNovelty(args.max_n)
    sizes, prop_unique = nov.get_novelty_lb(CORPUS_SIZE, entropy=ENTROPY, prob=P_AMBIGUOUS)
    idx = np.argmin(np.abs(prop_unique - 0.5))
    threshold = sizes[idx]
    print("LB threshold:", threshold)

    # Data used by all plots. 
    # data = CorpusData.load(args.root)
    lmg = LMGenerations.load(args.root)
    results = Results.load(args.root)
    lengths_12b = get_lengths_for_model("EleutherAI/pythia-12b")

    # Main plots.
    plot_model(args, "EleutherAI/pythia-12b")
    # plot_by_plen(args)
    # plot_frequency(args)

    # Model size and domain experiments.
    # plot_by_model(args)
    # plot_by_domain(args)
    # plot_by_domain(args, deduped=True)

    # Decoding experiments.
    # decoding = DecodingPlots(args)
    # decoding.plot_by_topk(lmg, results, lengths_12b)
    # decoding.plot_by_topp(lmg, results, lengths_12b)
    # decoding.plot_by_temp(lmg, results, lengths_12b)
    # decoding.plot_by_beam(lmg, results, lengths_12b)

    # Completion loss experiments.
    # closs = CompletionLossPlots(args)
    # closs.plot_by_model(results)
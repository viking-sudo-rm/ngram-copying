"""Script to regenerate all of the plots that we need."""

from collections import Counter, defaultdict
from typing import NamedTuple
import numpy as np
import argparse
import json
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.data import CorpusData, LMGenerations, Results

plt.rcParams.update({'font.size': 13})

MODELS = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b", "pythia-2.8b",
          "pythia-6.9b", "pythia-12b"]
PLENS = [1, 10, 100]

BLUES = plt.cm.Blues(np.linspace(0.2, 1, len(MODELS)))
ORANGES = plt.cm.Oranges(np.linspace(0.2, 1, len(PLENS)))

def flatten(lists):
    return [item for row in lists for item in row]

def clean_model_name(model):
    return model.replace("EleutherAI/", "")

def clean_size(size) -> float:
    if size.endswith("m"):
        return float(size[:-1]) * 1e6
    elif size.endswith("b"):
        return float(size[:-1]) * 1e9
    else:
        raise ValueError

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str,
                        default="/net/nfs.cirrascale/allennlp/willm/ngram-copying")
    parser.add_argument("--max_n", "-n", type=int, default=10)
    return parser.parse_args()

def get_proportion_unique(suffix_contexts, max_n: int = 10):
    """Convert length data to n-gram novelty data"""
    lengths = np.arange(max_n)
    counter = Counter(suffix_contexts)
    freqs = np.array([counter[l] for l in lengths])
    prop_unique = (np.cumsum(freqs) - lengths) / (-lengths + len(suffix_contexts))
    return lengths + 1, prop_unique

def format_novelty_plot(plt, max_n: int = 10):
    plt.legend()
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.xlabel("n-gram size")
    plt.ylabel("% novel")
    plt.xticks(list(range(1, max_n + 1)))
    plt.ylim([0, 1])
    plt.tight_layout()

def plot_model(args, model="EleutherAI/pythia-12b", name="Pythia-12B"):
    plot_lengths = defaultdict(list)
    plot_lengths["val"] = results.val_iid["lengths"]
    for doc, lengths in zip(lmg.by_model, results.by_model["lengths"]):
        if doc["meta"]["model"] != model:
            continue
        plen = doc["meta"]["prompt_len"]
        plot_lengths[plen].append(lengths)

    plt.figure()
    sizes, prop_unique = get_proportion_unique(flatten(plot_lengths["val"]), max_n=args.max_n)
    plt.plot(sizes, prop_unique, linestyle="--", color="gray", label="val")
    for color, plen in zip(ORANGES, PLENS):
        sizes, prop_unique = get_proportion_unique(flatten(plot_lengths[plen]), max_n=args.max_n)
        plt.plot(sizes, prop_unique, color=color, label=f"p={plen}")
    format_novelty_plot(plt, max_n=args.max_n)
    plt.title(f"n-gram novelty of {name}")

    os.makedirs("plots/by-model", exist_ok=True)
    plt.savefig(f"plots/by-model/{clean_model_name(model)}.pdf")
    plt.close()

def plot_by_model(args, plen=1):
    os.makedirs("plots/by-model", exist_ok=True)
    
    plot_lengths = defaultdict(list)
    plot_lengths["val"] = results.val_iid["lengths"]
    for doc, lengths in zip(lmg.by_model, results.by_model["lengths"]):
        if doc["meta"]["prompt_len"] != plen:
            continue
        model = clean_model_name(doc["meta"]["model"])
        plot_lengths[model].append(lengths)

    plt.figure()
    sizes, prop_unique = get_proportion_unique(flatten(plot_lengths["val"]), max_n=args.max_n)
    plt.plot(sizes, prop_unique, label="val", color="gray", linestyle="--")
    for color, key in zip(BLUES, MODELS):
        sizes, prop_unique = get_proportion_unique(flatten(plot_lengths[key]), max_n=args.max_n)
        plt.plot(sizes, prop_unique, label=key.split("-")[-1], color=color)
    format_novelty_plot(plt, max_n=args.max_n)
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
    plt.title("non-novel suffix len")
    plt.scatter(sizes, mean_lengths, linestyle="-")
    plt.xlabel("model size")
    plt.ylabel("mean non-novel suffix len")
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig("plots/by-model/scatter.pdf")
    plt.close()

def plot_by_domain(args):
    os.makedirs("plots/by-domain", exist_ok=True)
    lengths_by_domain = defaultdict(list)
    domains = set()

    # First compute for validation set.
    for doc, lengths in zip(data.val_by_domain, results.val_by_domain["lengths"]):
        domain = doc["meta"]["pile_set_name"]
        lengths_by_domain[domain, "val"].append(lengths)
        domains.add(domain)

    # Next compute for CDAWG outputs.
    for doc, lengths in zip(lmg.by_domain, results.by_domain["lengths"]):
        plen = doc["meta"]["prompt_len"]
        pidx = doc["meta"]["prompt_idx"]
        prompt = data.prompts_by_domain[pidx]
        domain = prompt["meta"]["pile_set_name"]
        lengths_by_domain[domain, plen].append(lengths)
        domains.add(domain)

    for domain in domains:
        plt.figure()
        val_lengths = lengths_by_domain[domain, "val"]
        val_sizes, val_prop_unique = get_proportion_unique(flatten(val_lengths), max_n=args.max_n)
        plt.plot(val_sizes, val_prop_unique, linestyle="--", color="gray")
        for color, plen in zip(ORANGES, PLENS):
            lengths = lengths_by_domain[domain, plen]
            sizes, prop_unique = get_proportion_unique(flatten(lengths), max_n=args.max_n)
            plt.plot(sizes, prop_unique, label=f"p={plen}", color=color)
        plt.title(f"n-gram novelty for {domain}")
        format_novelty_plot(plt, max_n=args.max_n)
        plt.savefig(f"plots/by-domain/{domain}.pdf")
        plt.close()

if __name__ == "__main__":
    args = parse_args()
    data = CorpusData.load(args.root)
    lmg = LMGenerations.load(args.root)
    results = Results.load(args.root)

    plot_model(args, model="EleutherAI/pythia-12b")
    plot_by_model(args)
    plot_by_domain(args)

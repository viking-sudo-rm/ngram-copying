"""Script to regenerate all of the plots that we need."""

from collections import Counter, defaultdict
from typing import NamedTuple
import numpy as np
import argparse
import json
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

COLORS = mcolors.CSS4_COLORS
plt.rcParams.update({'font.size': 13})

MODELS = ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
          "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
          "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b"]

class CorpusData(NamedTuple):
    prompts_iid: list
    val_iid: list
    prompts_by_domain: list
    val_by_domain: list

    @classmethod
    def load(cls, root):
        with open(os.path.join(root, "data/iid/prompts.jsonl")) as fh:
            prompts_iid = [json.loads(line) for line in fh]
        with open(os.path.join(root, "data/iid/val.jsonl")) as fh:
            val_iid = [json.loads(line) for line in fh]
        with open(os.path.join(root, "data/by-domain/prompts.jsonl")) as fh:
            prompts_by_domain = [json.loads(line) for line in fh]
        with open(os.path.join(root, "data/by-domain/val.jsonl")) as fh:
            val_by_domain = [json.loads(line) for line in fh]
        return cls(prompts_iid, val_iid, prompts_by_domain, val_by_domain)

class LMGenerations(NamedTuple):
    by_domain: list
    by_model: list
    pythia_12b: list

    @classmethod
    def load(cls, root):
        with open(os.path.join(root, "lm-generations/by-domain/pythia-12b.jsonl")) as fh:
            by_domain = [json.loads(line) for line in fh]
        with open(os.path.join(root, "lm-generations/by-model.jsonl")) as fh:
            by_model = [json.loads(line) for line in fh]
        with open(os.path.join(root, "lm-generations/by-model/pythia-12b.jsonl")) as fh:
            pythia_12b = [json.loads(line) for line in fh]
        return cls(by_domain, by_model, pythia_12b)

class Results(NamedTuple):
    val_by_domain: dict
    val_iid: dict
    by_domain: dict
    by_model: dict

    @classmethod
    def load(cls, root):
        with open(os.path.join(root, "results/val.json")) as fh:
            val_by_domain = json.load(fh)
        with open(os.path.join(root, "results/val-iid.json")) as fh:
            val_iid = json.load(fh)
        with open(os.path.join(root, "results/by-domain.json")) as fh:
            by_domain = json.load(fh)
        with open(os.path.join(root, "results/by-model.json")) as fh:
            by_model = json.load(fh)
        return cls(val_by_domain, val_iid, by_domain, by_model)

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

def plot_model(args, model="EleutherAI/pythia-12b"):
    plot_lengths = defaultdict(list)
    plot_lengths["val"] = results.val_iid["lengths"]
    for doc, lengths in zip(lmg.by_model, results.by_model["lengths"]):
        if doc["meta"]["model"] != model:
            continue
        plen = doc["meta"]["prompt_len"]
        plot_lengths[f"p={plen}"].append(lengths)

    plt.figure()
    for key, lengths in plot_lengths.items():
        sizes, prop_unique = get_proportion_unique(flatten(lengths), max_n=args.max_n)
        plt.plot(sizes, prop_unique, label=key)
    format_novelty_plot(plt, max_n=args.max_n)
    os.makedirs("plots/by-model", exist_ok=True)
    plt.savefig(f"plots/by-model/{clean_model_name(model)}.pdf")

def plot_by_model(args, plen=1):
    plt.figure(figsize=(10, 5))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.suptitle("n-gram novelty and non-novel suffix length by model size")

    blue_palette = plt.cm.Blues(np.linspace(0.2, 1, len(MODELS)))
    # baseline_orange = '#FFA500'
    
    plot_lengths = defaultdict(list)
    plot_lengths["val"] = results.val_iid["lengths"]
    for doc, lengths in zip(lmg.by_model, results.by_model["lengths"]):
        if doc["meta"]["prompt_len"] != plen:
            continue
        model = clean_model_name(doc["meta"]["model"])
        plot_lengths[model].append(lengths)

    sizes, prop_unique = get_proportion_unique(flatten(plot_lengths["val"]), max_n=args.max_n)
    ax1.plot(sizes, prop_unique, label="val", color="gray")
    for color, key in zip(blue_palette, MODELS):
        sizes, prop_unique = get_proportion_unique(flatten(plot_lengths[key]), max_n=args.max_n)
        ax1.plot(sizes, prop_unique, label=key.split("-")[-1], color=color)
    ax1.set_title("n-gram novelty")
    ax1.set(xlabel="n-gram size", ylabel="% novel")
    plt.sca(ax1)
    plt.xticks(list(range(1, args.max_n + 1)))
    plt.ylim([0, 1])
    plt.legend()
    
    sizes = []
    mean_lengths = []
    for model, lengths in plot_lengths.items():
        if model == "val":
            continue
        sizes.append(clean_size(model.split("-")[-1]))
        mean_lengths.append(np.mean(flatten(lengths)))

    ax2.set_title("non-novel suf len")
    ax2.scatter(sizes, mean_lengths, linestyle="-", color="gray")
    ax2.set(xlabel="model size", ylabel="mean non-novel suffix length")
    plt.sca(ax2)
    plt.xscale("log")

    fig.tight_layout()
    plt.savefig("plots/by-model.pdf")

def plot_by_domain(args, plen=1):
    print(f"=== Plotting with plen={plen} ===")

    # First compute for validation set.
    val_lengths_by_domain = defaultdict(list)
    for doc, lengths in zip(data.val_by_domain, results.val_by_domain["lengths"]):
        domain = doc["meta"]["pile_set_name"]
        val_lengths_by_domain[domain].append(lengths)
    print("# of val domains:", len(val_lengths_by_domain))

    # Next compute for CDAWG outputs.
    lengths_by_domain = defaultdict(list)
    for doc, lengths in zip(lmg.by_domain, results.by_domain["lengths"]):
        if doc["meta"]["prompt_len"] != plen:
            continue
        pidx = doc["meta"]["prompt_idx"]
        prompt = data.prompts_by_domain[pidx]
        domain = prompt["meta"]["pile_set_name"]
        lengths_by_domain[domain].append(lengths)
    print("# of domains:", len(lengths_by_domain))

    plt.figure()
    colors = list(COLORS.values())
    for color, (domain, lengths) in zip(colors, lengths_by_domain.items()):
        val_lengths = val_lengths_by_domain[domain]
        val_sizes, val_prop_unique = get_proportion_unique(flatten(val_lengths), max_n=args.max_n)
        plt.plot(val_sizes, val_prop_unique, linestyle="--", color=color)
        sizes, prop_unique = get_proportion_unique(flatten(lengths), max_n=args.max_n)
        plt.plot(sizes, prop_unique, label=domain, color=color)
    plt.title(f"n-gram novelty by domain, p={plen}")
    format_novelty_plot(plt, max_n=args.max_n)
    os.makedirs("plots/by-domain", exist_ok=True)
    plt.savefig(f"plots/by-domain/domain-{plen}.pdf")

if __name__ == "__main__":
    args = parse_args()
    data = CorpusData.load(args.root)
    lmg = LMGenerations.load(args.root)
    results = Results.load(args.root)

    plot_by_model(args)
    for model in MODELS:
        plot_model(args, model)

    # for plen in [1, 10, 100]:
    #     plot_by_domain(args, plen=1)

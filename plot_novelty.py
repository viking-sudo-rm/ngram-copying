"""Script to regenerate all of the plots that we need."""

from collections import Counter, defaultdict
from typing import NamedTuple
import numpy as np
import argparse
import json
import os
import matplotlib.pyplot as plt

PROMPT_LENGTHS = [1, 4, 16, 64, 100]

plt.rcParams.update({'font.size': 13})

class Data(NamedTuple):
    prompts: dict
    results_val: dict
    docs_12b: list
    results_12b: dict

    @classmethod
    def load(cls, root):
        with open(os.path.join(root, "prompts.jsonl")) as fh:
            prompts = [json.loads(line) for line in fh]
        with open(os.path.join(root, "results/val-20.json")) as fh:
            results_val = json.load(fh)
        with open(os.path.join(root, "lm-generations/models/pythia-12b.jsonl")) as fh:
            docs_12b = [json.loads(line) for line in fh]
        with open(os.path.join(root, "results/models/pythia-12b.json")) as fh:
            results_12b = json.load(fh)
        return cls(prompts, results_val, docs_12b, results_12b)

def flatten(lists):
    return [item for row in lists for item in row]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str,
                        default="/net/nfs.cirrascale/allennlp/willm/ngram-copying")
    parser.add_argument("--max_n", "-n", type=int, default=20)
    return parser.parse_args()

def get_proportion_unique(suffix_contexts, max_n: int = 10):
    """Convert length data to n-gram novelty data"""
    lengths = np.arange(max_n)
    counter = Counter(suffix_contexts)
    freqs = np.array([counter[l] for l in lengths])
    prop_unique = (np.cumsum(freqs) - lengths) / (-lengths + len(suffix_contexts))
    return lengths + 1, prop_unique

def format_novelty_plot(plt, max_n: int = 10, legend_loc="best"):
    plt.legend(loc=legend_loc)
    plt.xlabel("n-gram size")
    plt.ylabel("% novel")
    plt.xticks(list(range(1, max_n + 1)))
    plt.ylim([0, 1])
    plt.tight_layout()

def main_plot(args):
    
    lengths, prop_unique = get_proportion_unique(flatten(val["lengths"]))

    plt.figure()
    plt.plot(lengths, prop_unique, label="val")
    for plen in PROMPT_LENGTHS:
        lengths = [l for d, l in zip(pythia12b_docs, pythia12b["lengths"])
                   if d["meta"]["prompt_len"] == plen]
        sizes, prop_unique = get_proportion_unique(flatten(lengths))
        plt.plot(sizes, prop_unique, label=f"p={plen}")
    plt.title("pythia-12b n-gram novelty by prompt length")
    format_novelty_plot(plt)
    plt.savefig("plots/main.pdf")

def plot_by_domain(args, plen=1):
    lengths_by_domain = defaultdict(list)
    lengths_12b = data.results_12b["lengths"]
    for doc, lengths in zip(data.docs_12b, lengths_12b):
        if doc["meta"]["prompt_len"] != plen:
            continue
        pidx = doc["meta"]["prompt_idx"]
        prompt = data.prompts[pidx]
        domain = prompt["meta"]["pile_set_name"]
        lengths_by_domain[domain].append(lengths)

    plt.figure()
    lengths, prop_unique = get_proportion_unique(flatten(data.results_val["lengths"]), max_n=args.max_n)
    for domain, lengths in lengths_by_domain.items():
        sizes, prop_unique = get_proportion_unique(flatten(lengths), max_n=args.max_n)
        plt.plot(sizes, prop_unique, label=domain)
    plt.title(f"pythia-12b n-gram novelty by domain (p={plen})")
    format_novelty_plot(plt, max_n=args.max_n)
    os.makedirs("plots/domain", exist_ok=True)
    plt.savefig(f"plots/domain/domain-{plen}.pdf")

if __name__ == "__main__":
    args = parse_args()
    data = Data.load(args.root)
    plot_by_domain(args, plen=1)
    plot_by_domain(args, plen=4)
    plot_by_domain(args, plen=16)
    plot_by_domain(args, plen=64)
    plot_by_domain(args, plen=100)

"""Script to regenerate all of the plots that we need."""

from collections import Counter
import numpy as np
import argparse
import json
import os
import matplotlib.pyplot as plt

PROMPT_LENGTHS = [1, 4, 16, 64, 100]

plt.rcParams.update({'font.size': 13})

def flatten(lists):
    return [item for row in lists for item in row]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str,
                        default="/net/nfs.cirrascale/allennlp/willm/ngram-copying")
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
    with open(os.path.join(args.root, "results/val-20.json")) as fh:
        val = json.load(fh)
    lengths, prop_unique = get_proportion_unique(flatten(val["lengths"]))
    plt.plot(lengths, prop_unique, label="val")

    with open(os.path.join(args.root, "lm-generations/models/pythia-12b.jsonl")) as fh:
        pythia12b_docs = [json.loads(line) for line in fh]
    with open(os.path.join(args.root, "results/models/pythia-12b.json")) as fh:
        pythia12b = json.load(fh)
    for plen in PROMPT_LENGTHS:
        lengths = [l for d, l in zip(pythia12b_docs, pythia12b["lengths"])
                   if d["meta"]["prompt_len"] == plen]
        sizes, prop_unique = get_proportion_unique(flatten(lengths))
        plt.plot(sizes, prop_unique, label=f"p={plen}")

    plt.title("pythia-12b n-gram novelty by prompt length")
    format_novelty_plot(plt)
    plt.savefig("plots/main.pdf")

if __name__ == "__main__":
    args = parse_args()
    main_plot(args)

import argparse
import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from rusty_dawg import Cdawg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-12b")
    parser.add_argument("--gen_path", type=str, default="/net/nfs.cirrascale/allennlp/willm/ngram-copying/lm-generations/models.jsonl")
    parser.add_argument("--val_path", type=str, default="/net/nfs.cirrascale/allennlp/willm/ngram-copying/val.jsonl")
    parser.add_argument("--max_n", type=int, default=20)
    return parser.parse_args()

def get_in_context_suffix_lengths(tokens):
    """Uses a CDAWG to get in-context suffix lengths."""
    cdawg = Cdawg(tokens)
    lengths = []
    state, start = cdawg.get_source(), 1
    for idx, token in enumerate(tokens):
        state, start = cdawg.update(state, start, idx + 1)
        gamma = (start - 1, idx + 1)  # Inference (0-indexed) version
        cs = cdawg.implicitly_fail(state, gamma)
        # Can't just call cs.get_length() because -1 length for null not represented.
        fstate, fgamma = cs.get_state_and_gamma()
        fgamma = (fgamma[0] - 1, fgamma[1])
        fstate_length = -1 if fstate is None else cdawg.get_length(fstate)
        lengths.append(fstate_length + fgamma[1] - fgamma[0])
    return lengths

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

args = parse_args()

with open(args.gen_path) as fh:
    blobs = [json.loads(line) for line in fh.readlines()]

with open(args.val_path) as fh:
    val_blobs = [json.loads(line) for line in fh.readlines()]

tokenizer = AutoTokenizer.from_pretrained(args.model)

suffix_lengths_by_n = defaultdict(list)
for blob in blobs:
    meta = blob["meta"]
    # Decide whether to exclude no_repeat data here
    if meta["model"] != args.model or "no_repeat_ngram_size" in meta:
        continue
    n = meta["prompt_len"]
    suffix_lengths = get_in_context_suffix_lengths(blob["tokens"])
    suffix_lengths = suffix_lengths[n:]  # Remove prompt itself.
    suffix_lengths_by_n[n].extend(suffix_lengths)

print("# entries:", len(suffix_lengths_by_n[1]))

for blob in val_blobs:
    tokens = tokenizer.encode(blob["text"])
    suffix_lengths = get_in_context_suffix_lengths(tokens)
    suffix_lengths_by_n["val"].extend(suffix_lengths)

plt.figure()
lengths, novelty = get_proportion_unique(suffix_lengths_by_n["val"], max_n=args.max_n)
plt.plot(lengths, novelty, label=f"val")
for n in [1, 10, 100]:
    lengths, novelty = get_proportion_unique(suffix_lengths_by_n[n], max_n=args.max_n)
    plt.plot(lengths, novelty, label=f"n={n}")
format_novelty_plot(plt, max_n=args.max_n)
plt.savefig("plots/novelty.png")
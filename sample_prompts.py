from argparse import ArgumentParser
import random

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--path", default="/net/nfs.cirrascale/allennlp/willm/data/pile/00_0.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", default="/net/nfs.cirrascale/allennlp/willm/ngram-copying/prompts.jsonl")
    parser.add_argument("--n_prompts", type=int, default=20)
    return parser.parse_args()


args = parse_args()
random.seed(args.seed)
with open(args.path) as fh:
    lines = fh.readlines()
sample = random.sample(lines, args.n_prompts)
with open(args.save_path, "w") as fh:
    for line in sample:
        assert line.strip()
        fh.write(line.strip() + "\n")
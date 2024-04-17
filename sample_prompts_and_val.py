"""Sample prompts and validation examples from the Pile.

Maybe also add an option to do this with rejection sampling for rare classes."""

from argparse import ArgumentParser
import random
import json
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import Counter, defaultdict

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--train_path", default="/net/nfs.cirrascale/allennlp/willm/data/pile/00_0.json")
    parser.add_argument("--val_path", default="/net/nfs.cirrascale/allennlp/willm/data/pile/val.jsonl")
    parser.add_argument("--prompts_save_path", default="/net/nfs.cirrascale/allennlp/willm/ngram-copying/data/uniform/prompts.jsonl")
    parser.add_argument("--val_save_path", default="/net/nfs.cirrascale/allennlp/willm/ngram-copying/data/uniform/val.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("-t", "--tokenizer", type=str, default="EleutherAI/pythia-12b")
    parser.add_argument("--max_val_tokens", type=int, default=1000)
    parser.add_argument("--by_domain", action="store_true")
    return parser.parse_args()

class Jsonl:
    def __init__(self, lines):
        self.lines = lines

    @classmethod
    def sample_from(cls, path: str, n_samples: int):
        with open(path) as fh:
            lines = fh.readlines()
        sampled_lines = random.sample(lines, n_samples)
        assert all(line.strip() for line in sampled_lines)
        return cls(sampled_lines)
    
    @classmethod
    def sample_by_domain(cls, path: str, n_samples: int):
        with open(path) as fh:
            lines = fh.readlines()
        blobs = [json.loads(line) for line in lines]

        by_domain = defaultdict(list)
        for blob in blobs:
            domain = blob["meta"]["pile_set_name"]
            by_domain[domain].append(blob)

        sampled_lines = []
        for domain, blobs in by_domain.items():
            if n_samples < len(blobs):
                blobs = random.sample(blobs, n_samples)
            sampled_lines.extend(json.dumps(blob) for blob in blobs)
        
        return cls(sampled_lines)
    
    def get_domain_counts(self):
        blobs = [json.loads(line) for line in self.lines]
        domains = [blob["meta"]["pile_set_name"] for blob in blobs]
        return Counter(domains)

    def tokenize_and_trim(self, tokenizer, max_tokens: int):
        blobs = [json.loads(line) for line in self.lines]
        tokens = [tokenizer.encode(blob["text"]) for blob in blobs]
        metas = [blob["meta"] for blob in blobs]
        blobs = [{"tokens": t[:max_tokens], "meta": m} for t, m in zip(tqdm(tokens), metas)]
        self.lines = [json.dumps(blob) for blob in blobs]

    def save(self, path):
        with open(path, "w") as fh:
            for line in self.lines:
                fh.write(line.strip() + "\n")

def main(args):
    random.seed(args.seed)

    print("Sampling and saving prompts...")
    if args.by_domain:
        prompts = Jsonl.sample_by_domain(args.train_path, args.n_samples)
    else:
        prompts = Jsonl.sample_from(args.train_path, args.n_samples)
    prompts.save(args.prompts_save_path)
    print(prompts.get_domain_counts())

    print("\nSampling and saving validation documents...")
    if args.by_domain:
        val = Jsonl.sample_by_domain(args.val_path, args.n_samples)
    else:
        val = Jsonl.sample_from(args.val_path, args.n_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    val.tokenize_and_trim(tokenizer, args.max_val_tokens)
    val.save(args.val_save_path)
    print(val.get_domain_counts())

if __name__ == "__main__":
    main(parse_args())
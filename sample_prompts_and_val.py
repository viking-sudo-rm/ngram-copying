"""Sample prompts and validation examples from the Pile.

Maybe also add an option to do this with rejection sampling for rare classes."""

from argparse import ArgumentParser
import random
import json
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import Counter, defaultdict
from datetime import datetime

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--train_path", default=None)
    parser.add_argument("--val_path", default=None)
    parser.add_argument("--prompts_save_path", default=None)
    parser.add_argument("--val_save_path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("-t", "--tokenizer", type=str, default="EleutherAI/pythia-12b")
    parser.add_argument("--max_val_tokens", type=int, default=1000)
    parser.add_argument("--by_domain", action="store_true")
    parser.add_argument("--format", choices=DOMAIN_KEYS.keys(), default="pile")
    parser.add_argument("--date_cutoff", type=str, default=None, help="ISO format, e.g., 2019-12-31T11:59:59+00:00")
    parser.add_argument("--date_key", type=str, default="created")
    parser.add_argument("--trim_date", action="store_true")
    return parser.parse_args()

DOMAIN_KEYS = {
    "pile": "pile_set_name",
    "cosmopedia": "split",
    "dolma": "subdomain",
}

class Jsonl:
    def __init__(self, lines, fmt):
        self.lines = lines
        self.fmt = fmt

    @classmethod
    def load(cls, path: str, **kwargs):
        with open(path) as fh:
            lines = fh.readlines()
        return cls(lines, **kwargs)

    def sample_from(self, n_samples: int):
        if len(self.lines) < n_samples:
            print(f"WARNING: Didn't sample because # samples ({n_samples}) < # lines ({len(self.lines)})")
            return
        self.lines = random.sample(self.lines, n_samples)
        assert all(line.strip() for line in self.lines)
    
    def sample_by_domain(self, n_samples: int):
        blobs = [json.loads(line) for line in self.lines]
        by_domain = defaultdict(list)
        for blob in blobs:
            domain = self.get_domain(blob)
            by_domain[domain].append(blob)

        self.lines = []
        for domain, blobs in by_domain.items():
            if n_samples < len(blobs):
                blobs = random.sample(blobs, n_samples)
            self.lines.extend(json.dumps(blob) for blob in blobs)

    def get_domain_counts(self):
        blobs = [json.loads(line) for line in self.lines]
        domains = [self.get_domain(blob) for blob in blobs]
        return Counter(domains)

    def tokenize_and_trim(self, tokenizer, max_tokens: int):
        blobs = [json.loads(line) for line in self.lines]
        tokens = [tokenizer.encode(blob["text"]) for blob in blobs]
        metas = [self.get_meta(blob) for blob in blobs]
        blobs = [{"tokens": t[:max_tokens], "meta": m} for t, m in zip(tqdm(tokens), metas)]
        self.lines = [json.dumps(blob) for blob in blobs]

    def save(self, path):
        with open(path, "w") as fh:
            for line in self.lines:
                fh.write(line.strip() + "\n")

    def get_meta(self, blob):
        match self.fmt:
            case "pile":
                return blob["meta"]
            case "cosmopedia" | "dolma":
                return {k: v for k, v in blob.items() if k != "text"}
            case _:
                raise ValueError(f"Unknown format {self.fmt}")

    def get_domain(self, blob):
        meta = self.get_meta(blob)
        key = DOMAIN_KEYS[self.fmt]
        return meta[key]

    def filter(self, filter_fn):
        self.lines = [line for line in self.lines if filter_fn(line)]
    
    def filter_by_date(self, date, key="created", trim=False):
        """Date is string in ISO format and key field should exist."""
        date = datetime.fromisoformat(date)

        def _compare_fn(line):
            blob = json.loads(line)
            date_string = blob[key]
            if trim:
                date_string = date_string.split("T")[0]
            other_date = datetime.fromisoformat(date_string)
            return date < other_date

        self.filter(_compare_fn)
    
    def __len__(self):
        return len(self.lines)

def main(args):
    random.seed(args.seed)

    if args.train_path is not None:
        print("\nLoading prompts...")
        prompts = Jsonl.load(args.train_path, fmt=args.format)
        if args.date_cutoff is not None:
            print("Filtering by cutoff", args.date_cutoff)
            prompts.filter_by_date(args.date_cutoff, key=args.date_key, trim=args.trim_date)
            print("# lines after date filter:", len(prompts))
        print("Sampling and saving prompts...")
        if args.by_domain:
            prompts.sample_by_domain(args.n_samples)
        else:
            prompts.sample_from(args.n_samples)
        prompts.save(args.prompts_save_path)
        print(prompts.get_domain_counts())

    if args.val_path is not None:
        print("\nLoading validation text...")
        val = Jsonl.load(args.val_path, fmt=args.format)
        if args.date_cutoff is not None:
            print("Filtering by cutoff", args.date_cutoff)
            val.filter_by_date(args.date_cutoff, key=args.date_key, trim=args.trim_date)
            print("# lines after date filter:", len(val))
        print("\nSampling and saving validation documents...")
        if args.by_domain:
            val.sample_by_domain(args.n_samples)
        else:
            val.sample_from(args.n_samples)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        val.tokenize_and_trim(tokenizer, args.max_val_tokens)
        val.save(args.val_save_path)
        print("Documents saved!")
        print(val.get_domain_counts())

if __name__ == "__main__":
    main(parse_args())
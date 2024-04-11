"""Tokenize the validation set and only keep the first 1000 tokens from each document."""

from tqdm import tqdm
from transformers import AutoTokenizer
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("-t", "--tokenizer", type=str, default="EleutherAI/pythia-12b")
    parser.add_argument("--max_tokens", type=int, default=1000)
    return parser.parse_args()

def main(args):
    with open(args.input_path) as fh:
        blobs = [json.loads(line) for line in fh]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)    
    tokens = [tokenizer.encode(blob["text"]) for blob in blobs]
    metas = [blob["meta"] for blob in blobs]
    blobs = [{"tokens": t[:args.max_tokens], "meta": m} for t, m in zip(tqdm(tokens), metas)]

    with open(args.output_path, "w") as fh:
        fh.write("\n".join(json.dumps(blob) for blob in blobs))

if __name__ == "__main__":
    main(parse_args())
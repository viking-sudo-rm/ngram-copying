"""Convert model-generated text (JSONL format) to CSV-formatted documents for visualization."""

import json
import pandas as pd
import argparse
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_path", type=str)
    parser.add_argument("csv_path", type=str)
    parser.add_argument("--tokenizer", "-t", type=str, default="EleutherAI/pythia-12b")
    return parser.parse_args()

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    with open(args.jsonl_path) as fh:
        lines = [json.loads(line) for line in fh]
    breakpoint()
    df = pd.DataFrame({
        "model": [line["meta"]["model"] for line in lines],
        "prompt": [tokenizer.decode(line["prompt"]) for line in lines],
        "text": [tokenizer.decode(line["tokens"]) for line in lines],
        "prompt len": [len(line["prompt"]) for line in lines],
        "text len": [len(line["tokens"]) for line in lines],
        "decoding": [json.dumps(line["meta"]["decoding"]) for line in lines],
    })
    df.to_csv(args.csv_path)

if __name__ == "__main__":
    main(parse_args())
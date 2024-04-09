"""Python script to pass text at a file through the CDAWGs over the JSON API.

Example usage:
python query_cdawgs.py \
    /net/nfs.cirrascale/allennlp/willm/ngram-copying/gen.jsonl \
    /net/nfs.cirrascale/allennlp/willm/ngram-copying/results/gen.json
"""

import argparse
import json
from src.async_client import AsyncRustyDawgClient

MACHINES = [f"localhost:{port}" for port in range(5000, 5030)]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("text_path", type=str)
    parser.add_argument("save_path", type=str)
    return parser.parse_args()

def main(args):
    # Assumes the input is in .jsonl format with a "text" field.
    with open(args.text_path, "r") as fh:
        blobs = [json.loads(line) for line in fh]
        texts = [blob["text"] for blob in blobs]
    
    # Create a client to pass text through the API.
    client = AsyncRustyDawgClient(MACHINES)
    results = client.query(texts)

    # Save results in .json format to the specified file. Document order will be the same as in input file.
    with open(args.save_path, "w") as fh:
        print(json.dumps(results, fh))

if __name__ == "__main__":
    main(parse_args())

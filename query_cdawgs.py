"""Python script to pass text at a file through the CDAWGs over the JSON API.

Example usage:
python query_cdawgs.py \
    /net/nfs.cirrascale/allennlp/willm/ngram-copying/gen.jsonl \
    /net/nfs.cirrascale/allennlp/willm/ngram-copying/results/gen.json
"""

import argparse
import json
import asyncio
from src.async_client import AsyncRustyDawgClient

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("text_path", type=str)
    parser.add_argument("save_path", type=str)
    parser.add_argument("--start_port", type=int, default=5000)
    parser.add_argument("--end_port", type=int, default=5030)
    return parser.parse_args()

async def main(args):
    # Assumes the input is in .jsonl format with a "text" field.
    with open(args.text_path, "r") as fh:
        blobs = [json.loads(line) for line in fh]
        texts = [blob["text"] for blob in blobs]
    print("Texts:", texts)

    # Create a client to pass text through the API.
    hosts = [f"localhost:{port}" for port in range(args.start_port, args.end_port)]
    print("Hosts:", hosts)
    client = AsyncRustyDawgClient(hosts)
    results = await client.query(texts)

    # Save results in .json format to the specified file. Document order will be the same as in input file.
    with open(args.save_path, "w") as fh:
        json.dump(results, fh)

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))

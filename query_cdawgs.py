"""Python script to pass text at a file through the CDAWGs over the JSON API.

Example usage:
python query_cdawgs.py \
    /net/nfs.cirrascale/allennlp/willm/ngram-copying/gen.jsonl \
    /net/nfs.cirrascale/allennlp/willm/ngram-copying/results/gen.json
"""

from tqdm import trange
import argparse
import json
import asyncio
from collections import defaultdict

from src.async_client import AsyncRustyDawgClient

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("text_path", type=str)
    parser.add_argument("save_path", type=str)
    parser.add_argument("--text", action="store_true", help="Use text instead of tokens for input.")
    parser.add_argument("--start_port", type=int, default=5000)
    parser.add_argument("--end_port", type=int, default=5030)
    parser.add_argument("--batch_size", "-b", type=int, default=10, help="# of docs to send through API at once")
    parser.add_argument("--read_timeout", "-t", type=float, default=60.)
    return parser.parse_args()

def get_json(args, blobs):
    if args.text:
        texts = [blob["text"] for blob in blobs]
        return {"text": texts}
    else:
        tokens = []
        for blob in blobs:
            tokens.append(blob["tokens"])
            # Remove the prompt from the text to query.
            if "prompt" in blob:
                prompt_len = len(blob["prompt"])
                tokens[-1] = tokens[-1][prompt_len:]
        return {"tokens": tokens}

async def main(args):
    # Assumes the input is in .jsonl format with a "text" field.
    with open(args.text_path, "r") as fh:
        blobs = [json.loads(line) for line in fh]

    hosts = [f"localhost:{port}" for port in range(args.start_port, args.end_port)]
    print("Hosts:", hosts)
    results = defaultdict(list)
    for b in trange(0, len(blobs), args.batch_size):
        batch = blobs[b:b + args.batch_size]
        api_blob = get_json(args, batch)
        client = AsyncRustyDawgClient(hosts, read_timeout=args.read_timeout)
        batch_results = await client.query(api_blob)
        for key, values in batch_results.items():
            results[key].extend(values)

    # Save results in .json format to the specified file. Document order will be the same as in input file.
    with open(args.save_path, "w") as fh:
        json.dump(dict(results), fh)

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))

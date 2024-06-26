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
    parser.add_argument("--start-port", type=int, default=5000)
    parser.add_argument("--end-port", type=int, default=5030)
    parser.add_argument("--batch-size", "-b", type=int, default=10, help="# of docs to send through API at once")
    parser.add_argument("--read-timeout", "-t", type=float, default=60.)
    parser.add_argument("--n-tries", type=int, default=10)
    parser.add_argument("--return-entropies", action="store_true")
    parser.add_argument("--return-next-tokens", type=int, default=0)
    return parser.parse_args()

def get_json(args, blobs):
    kwargs = {}
    if args.return_entropies:
        kwargs["return_entropies"] = True
    if args.return_next_tokens != 0:
        kwargs["return_next_tokens"] = args.return_next_tokens

    if args.text:
        texts = [blob["text"] for blob in blobs]
        return {"text": texts, **kwargs}
    else:
        tokens = []
        for blob in blobs:
            tokens.append(blob["tokens"])
            # Note: VLLM doesn't include the prompt in the generated text!
            # if "prompt" in blob:
            #     prompt_len = len(blob["prompt"])
            #     tokens[-1] = tokens[-1][prompt_len:]
        return {"tokens": tokens, **kwargs}

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
        batch_results = await client.query(api_blob, n_tries=args.n_tries)
        for key, values in batch_results.items():
            results[key].extend(values)

        # Save intermediate results.
        with open(args.save_path, "w") as fh:
            json.dump(dict(results), fh)

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))

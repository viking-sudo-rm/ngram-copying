"""Generate text from a Huggingface checkpoint

Reference for model generation: https://huggingface.co/blog/how-to-generate
Reference for Pythia checkpoints: https://huggingface.co/EleutherAI/pythia-6.9b
"""

import tqdm
import json
import torch
from argparse import ArgumentParser
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import set_seed

def get_params_grid(args):
    all_params = {}
    if args.sample:
        all_params["sample-norepeat=2"] = dict(do_sample=True, no_repeat_ngram_size=2)
        all_params["sample"] = dict(do_sample=True)  # This does top_k=50 by default I believe.
    for b in args.beam:
        all_params[f"beam={b}-norepeat=2"] = dict(num_beams=b, no_repeat_ngram_size=2)
        all_params[f"beam={b}"] = dict(num_beams=b)
    for k in args.top_k:
        all_params[f"top_k={k}-norepeat=2"] = dict(do_sample=True, top_k=k, no_repeat_ngram_size=2)
        all_params[f"top_k={k}"] = dict(do_sample=True, top_k=k)
    for p in args.top_p:
        all_params[f"top_p={p}-norepeat=2"] = dict(do_sample=True, top_p=p, no_repeat_ngram_size=2)
        all_params[f"top_p={p}"] = dict(do_sample=True, top_p=p)
    for temp in args.temperature:
        all_params[f"temp={temp}-norepeat=2"] = dict(do_sample=True, temperature=temp, no_repeat_ngram_size=2)
        all_params[f"temp={temp}"] = dict(do_sample=True, temperature=temp)
    return all_params

class Generator:
    def __init__(self, tokenizer, model, params: dict, device="cuda", seed: int = 42):
        self.tokenizer = tokenizer
        self.model = model
        self.params = params
        self.device = device
        self.seed = seed

    @torch.no_grad()
    def iterate_generate(self,
                         input_ids,
                         full_length: int,
                         context_length: int = 512,
                         stride: int = 512,
                         n_return: int = 1,
                        ):
        set_seed(self.seed)
        # input_ids = tokenizer.encode("The", return_tensors="pt")
        while input_ids.size(1) < full_length:
            context = input_ids[:, -context_length:].to(self.device)
            num_return_sequences = n_return if context.size(0) == 1 else 1
            output_ids = self.model.generate(context,
                                             max_new_tokens=stride,
                                             pad_token_id=50256,
                                             num_return_sequences=num_return_sequences,
                                             **self.params)
            if input_ids.size(0) == output_ids.size(0):
                input_ids = torch.cat([input_ids[:, :-context_length], output_ids.cpu()], dim=1)
            else: # Only reachable on first batch.
                input_ids = output_ids.cpu()
        return input_ids

def append_to_jsonl(fh, prompts, all_tokens, texts, seed, model, decoding, plen):    
    for pidx, (prompt, tokens, text) in enumerate(zip(prompts, all_tokens, texts)):
        blob = {
            "prompt": prompt,
            "tokens": tokens.tolist(),
            "text": text,
            "meta": {
                "model": model,
                "decoding": decoding,
                "seed": seed,
                "prompt_idx": pidx,
                "prompt_len": plen,
            }
        }
        fh.write(json.dumps(blob))
        fh.write("\n")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("prompts_path", type=str)
    parser.add_argument("save_path", type=str)
    parser.add_argument("--n_tokens", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=10)

    # Arguments for grid search
    parser.add_argument("--sample", action="store_true",
                        help="Try decoding with naive sampling")
    parser.add_argument("-t", "--temperature", type=float, nargs="+", default=[],
                        help="List of temperatures to decode with")
    parser.add_argument("-k", "--top_k", type=int, nargs="+", default=[],
                        help="top-k sampling parameter list")
    parser.add_argument("-p", "--top_p", type=float, nargs="+", default=[],
                        help="top-p/nucleus sampling parameter list")
    parser.add_argument("-b", "--beam", type=int, nargs="+", default=[],
                        help="Beam sizes for argmax decoding")
    parser.add_argument("-n", "--prompt_lengths", type=int, nargs="+", default=[1, 10, 100],
                        help="Prefix length of prompts to use, in tokens")
    return parser.parse_args()

def main(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.cuda()

    # Get all parameters to explore.
    all_params = get_params_grid(args)

    # Load all the prompts from JSONL.
    with open(args.prompts_path) as fh:
        blobs = [json.loads(line) for line in fh.readlines()]
        prompts = [tokenizer.encode(blob["text"]) for blob in blobs]
    args.batch_size = min(args.batch_size, len(prompts))

    pbar = tqdm.tqdm(total=len(all_params) * len(prompts) * len(args.prompt_lengths))
    with open(args.save_path, "w") as fh:
        for plen in args.prompt_lengths:
            prefixes = torch.tensor([prompt[:plen] for prompt in prompts])
            for name, params in all_params.items():
                pbar.set_description(name)
                generator = Generator(tokenizer, model, params, args.device, args.seed)
                all_input_ids = []
                for b in range(0, prefixes.size(0), args.batch_size):
                    batch = prefixes[b:b + args.batch_size]
                    input_ids = generator.iterate_generate(batch, args.n_tokens).squeeze()
                    all_input_ids.extend(input_ids)
                    pbar.update(input_ids.size(0))
                texts = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in all_input_ids]
                append_to_jsonl(fh, prefixes.tolist(), all_input_ids, texts,
                                seed=args.seed,
                                model=args.model,
                                decoding=params,
                                plen=plen,)
    pbar.close()

if __name__ == "__main__":
    main(parse_args())

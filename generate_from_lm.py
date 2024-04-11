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
    """Get decoding parameters.

    See here: https://huggingface.co/docs/transformers/en/main_classes/text_generation
    """
    all_params = {}

    # === Greedy decoding methods ===
    if args.greedy:
        all_params["greedy"] = dict()
    # Greedy decoding with beam search.
    for b in args.beam:
        all_params[f"beam={b}"] = dict(num_beams=b)
    # Greedy decoding with no repetition.
    for r in args.no_repeat:
        all_params[f"no_repeat={r}"] = dict(no_repeat_ngram_size=r)

    # === Sampling decoding methods ===    
    if args.sample:
        all_params["sample"] = dict(do_sample=True)
    # Sampling with top k.
    for k in args.top_k:
        all_params[f"top_k={k}"] = dict(do_sample=True, top_k=k)
    # Nucleus sampling.
    for p in args.top_p:
        all_params[f"top_p={p}"] = dict(do_sample=True, top_p=p)
    # Modifying the temperature.
    for temp in args.temperature:
        all_params[f"temp={temp}"] = dict(do_sample=True, temperature=temp)

    return all_params

class Generator:
    def __init__(self, model, params: dict, device="cuda"):
        self.model = model
        self.params = params
        self.device = device

    @torch.no_grad()
    def generate(self, prompts, length: int):
        """Take some tokens and continue them."""
        prompts = prompts.to(self.device)
        mask = torch.ones_like(prompts)
        tokens = self.model.generate(input_ids=prompts,
                                     attention_mask=mask,
                                     max_new_tokens=length,
                                     **self.params)
        return self.remove_padding(tokens.tolist())

    def remove_padding(self, all_tokens):
        pad_token = self.model.pad_token_id
        for idx in range(len(all_tokens)):
            all_tokens[idx] = [t for t in all_tokens[idx] if t != pad_token]
        return all_tokens

def append_to_jsonl(fh, prompts, all_tokens: list, texts, model, decoding, plen):    
    for pidx, (prompt, tokens, text) in enumerate(zip(prompts, all_tokens, texts)):
        blob = {
            "prompt": prompt,
            "tokens": tokens,
            "text": text,
            "meta": {
                "model": model,
                "decoding": decoding,
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
    parser.add_argument("-n", "--prompt_lengths", type=int, nargs="+", default=[1, 4, 16, 64, 100],
                        help="Prefix length of prompts to use, in tokens")

    # === Decoding options ===
    parser.add_argument("--greedy", action="store_true", help="Try greedy decoding")
    parser.add_argument("--sample", action="store_true", help="Try naive sampling decoding")
    parser.add_argument("-t", "--temperature", type=float, nargs="+", default=[],
                        help="List of temperatures to decode with")
    parser.add_argument("-k", "--top_k", type=int, nargs="+", default=[],
                        help="top-k sampling parameter list")
    parser.add_argument("-p", "--top_p", type=float, nargs="+", default=[],
                        help="top-p/nucleus sampling parameter list")
    parser.add_argument("-b", "--beam", type=int, nargs="+", default=[],
                        help="Beam sizes for argmax decoding")
    parser.add_argument("-r", "--no_repeat", type=int, nargs="+", default=[],
                        help="N-gram lengths to prevent decoding.")
    return parser.parse_args()

def main(args):
    set_seed(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.cuda()
    model.pad_token_id = 0

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
                generator = Generator(model, params, args.device)
                all_input_ids = []
                for b in range(0, prefixes.size(0), args.batch_size):
                    batch = prefixes[b:b + args.batch_size]
                    input_ids = generator.generate(batch, args.n_tokens)
                    all_input_ids.extend(input_ids)
                    pbar.update(len(input_ids))
                texts = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in all_input_ids]
                append_to_jsonl(fh, prefixes.tolist(), all_input_ids, texts,
                                model=args.model,
                                decoding=params,
                                plen=plen,)
    pbar.close()
    print(tokenizer.decode(all_input_ids[0]))

if __name__ == "__main__":
    main(parse_args())

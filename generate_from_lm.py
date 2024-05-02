"""Generate text from a Huggingface checkpoint

Reference for model generation: https://huggingface.co/blog/how-to-generate
Reference for Pythia checkpoints: https://huggingface.co/EleutherAI/pythia-6.9b
"""

import tqdm
import json
import torch
from argparse import ArgumentParser
import os
import logging

from vllm import LLM, SamplingParams

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def get_params_grid(args) -> dict[str, SamplingParams]:
    """Get decoding parameters.

    Uses VLLM: https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
    Before, used HF: https://huggingface.co/docs/transformers/en/main_classes/text_generation
    """
    kwargs = dict(max_tokens=args.n_tokens)
    all_params = {}

    # === Sampling decoding methods ===    
    if args.sample:
        all_params["sample"] = SamplingParams(**kwargs)
    # Sampling with top k.
    for k in args.top_k:
        all_params[f"top_k={k}"] = SamplingParams(top_k=k, **kwargs)
    # Nucleus sampling.
    for p in args.top_p:
        all_params[f"top_p={p}"] = SamplingParams(top_p=p, **kwargs)
    # Modifying the temperature.
    for temp in args.temperature:
        all_params[f"temp={temp}"] = SamplingParams(temperature=temp, **kwargs)

    # === Beam search is a bit special ===
    for b in args.beam:
        # See https://github.com/vllm-project/vllm/issues/975
        all_params[f"beam={b}"] = SamplingParams(temperature=0., use_beam_search=True, n=b, **kwargs)

    return all_params

def write_jsonl(fh, model, all_outputs: dict):
    for (decoding, plen), outputs in all_outputs.items():
        for pidx, output in enumerate(outputs):
            for completion in output.outputs:
                blob = {
                    "prompt": output.prompt_token_ids,
                    "tokens": completion.token_ids,
                    "text": completion.text,
                    "meta": {
                        "model": model,
                        "decoding": decoding,
                        "prompt_idx": pidx,
                        "prompt_len": plen,
                    }
                }
                fh.write(json.dumps(blob))
                fh.write("\n")

def filter_null(prompts):
    original_len = len(prompts)
    prompts = [p for p in prompts if len(p) != 0]
    if len(prompts) != original_len:
        log.warning(f"Filtered {original_len - len(prompts)} null prompts")
    return prompts

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("prompts_path", type=str)
    parser.add_argument("save_path", type=str)
    parser.add_argument("--n_tokens", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt_lengths", type=int, nargs="+", default=[0],
                        help="Prefix lengths of prompts to use, in tokens")
    parser.add_argument("--one_prompt", action="store_true",
                        help="Only use one prompt. Add if p=0 and deterministic.")

    # === Decoding options ===
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
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LLM(model=args.model, seed=args.seed)
    tokenizer = model.get_tokenizer()
    eos = tokenizer.eos_token
    all_params = get_params_grid(args)

    # Load all the prompts from JSONL.
    with open(args.prompts_path) as fh:
        blobs = [json.loads(line) for line in fh.readlines()]
        full_prompts = [tokenizer.encode(blob["text"]) for blob in blobs]
        full_prompts = filter_null(full_prompts)

    all_outputs = {}
    for plen in args.prompt_lengths:
        for decoding, params in all_params.items():
            if plen != 0:
                prompts = [p[:plen] for p in full_prompts]
                outputs = model.generate(prompt_token_ids=prompts, sampling_params=params)
            elif args.one_prompt:
                outputs = model.generate([eos])
            else:
                outputs = model.generate([eos for _ in full_prompts])
            
            all_outputs[decoding, plen] = outputs

    with open(args.save_path, "w") as fh:
        write_jsonl(fh, args.model, all_outputs)


if __name__ == "__main__":
    main(parse_args())

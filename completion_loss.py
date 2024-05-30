from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("text_path", type=str)
    parser.add_argument("save_path", type=str)
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-12b")
    return parser.parse_args()

@torch.no_grad()
def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(args.device)

    with open(args.text_path) as fh:
        blobs = [json.loads(line) for line in fh.readlines()]

    losses = []
    for blob in tqdm(blobs):
        input_ids = torch.tensor(blob["tokens"], device=args.device, dtype=torch.long).unsqueeze(0)
        inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        outputs = model(**inputs)

        logits = outputs["logits"][..., :-1, :].contiguous()
        labels = inputs["input_ids"][..., 1:].contiguous()
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        losses.append(loss.tolist())

    with open(args.save_path, "w") as fh:
        blob = {
            "text_path": args.text_path,
            "model": args.model,
            "losses": losses,
        }
        json.dump(blob, fh)
    print(f"Saved results to {args.save_path}")

if __name__ == "__main__":
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
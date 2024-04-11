# N-gram Copying Experiments

## Usage

### Sampling Prompts

Prompts can be sampled from the Pile via:
```
python sample_prompts.py
```

### Preprocess Validation Set

To preprocess the validation set (tokenize and only keep first 1000 tokens from each document):
```bash
python process_val_data.py /net/nfs.cirrascale/allennlp/willm/ngram-copying/validation/val-20.jsonl /net/nfs.cirrascale/allennlp/willm/ngram-copying/validation/val-20-tokens.jsonl
```

### Generating Data from Models

To just run across all models, can do:

```bash
export ROOT=/net/nfs.cirrascale/allennlp/willm/ngram-copying
./generate_from_lms.sh
```

<details>
<summary>Example of how to generate from a specific model</summary>

```bash
MODEL=pythia-70m-deduped
python generate_from_lm.py \
    EleutherAI/${MODEL} \
    /net/nfs.cirrascale/allennlp/willm/ngram-copying/prompts.jsonl \
    /net/nfs.cirrascale/allennlp/willm/ngram-copying/gen.jsonl \
    --sample
```

Models: 70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b, 12b  ([more information](https://huggingface.co/EleutherAI/pythia-6.9b))
</details>

### Feeding Generated Data Through the API

```bash
python query_cdawgs.py \
    /net/nfs.cirrascale/allennlp/willm/ngram-copying/gen.jsonl \
    /net/nfs.cirrascale/allennlp/willm/ngram-copying/results/gen.json
```
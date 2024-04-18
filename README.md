# N-gram Copying Experiments

## Usage

First set the environment variable:
```bash
export ROOT=/net/nfs.cirrascale/allennlp/willm/ngram-copying
```

### Sampling Prompts and Validation Data

Prompts can be sampled from the Pile via:
```
python sample_prompts_and_val.py --n_samples=10000
```

Or to generate models with 100 samples per domain (besides one where there are not 100):
```
mkdir /net/nfs.cirrascale/allennlp/willm/ngram-copying/data/by-domain
python sample_prompts_and_val.py --by_domain --n_samples=100 \
    --prompts_save_path=$ROOT/data/by-domain/prompts.jsonl \
    --val_save_path=$ROOT/data/by-domain/val.jsonl
```

### Generating Data from Models

To generate data from each model:

```bash
scripts/generate_from_lms.sh
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
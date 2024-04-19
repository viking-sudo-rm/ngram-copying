# N-gram Copying Experiments

## Usage

First set the environment variable:
```bash
export ROOT=/net/nfs.cirrascale/allennlp/willm/ngram-copying
```

### Sampling Prompts and Validation Data

Prompts can be sampled from the Pile via:
```
python sample_prompts_and_val.py --n_samples=500
```

Or to generate models with 100 samples per domain (besides one where there are not 100):
```
mkdir /net/nfs.cirrascale/allennlp/willm/ngram-copying/data/by-domain
python sample_prompts_and_val.py --by_domain --n_samples=100 \
    --prompts_save_path=$ROOT/data/by-domain/prompts.jsonl \
    --val_save_path=$ROOT/data/by-domain/val.jsonl
```

### Generating Data from Models

To generate by domain (only use Pythia-12b though):
```bash
scripts/generate_by_model.sh by-domain by-domain
```

To do the same for deduped (only use Pythia-12b though):
```bash
SUFFIX="-deduped" scripts/generate_by_model.sh by-domain by-domain-deduped
```

To generate by model (from IID data):
```bash
scripts/generate_by_model.sh iid by-model
```

To generate by decoding:
```bash
scripts/generate_by_decoding.sh
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

### Generating with Varying Decoding Strategies

```bash
scripts/generate_by_decoding.sh
```

### Feeding Generated Data Through the API

```bash
python query_cdawgs.py \
    $ROOT/lm-generations/models.jsonl \
    $ROOT/results/models.json
```
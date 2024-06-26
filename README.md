# N-gram Copying Experiments

## Motivation

This library uses CDAWGs (data structures for fast search in massive text corpora) to study the copying behavior of large language models. Specifically, we use a CDAWG built on the Pile to study the copying behavior of the Pythia models.

## Experiments on the Pile

First set the environment variable:
```bash
export ROOT=/net/nfs.cirrascale/allennlp/willm/ngram-copying
```

### Sampling Prompts and Validation Data

Prompts can be sampled from the Pile via:
```shell
python sample_prompts_and_val.py \
    --n_samples 500 \
    --train_path /net/nfs.cirrascale/allennlp/willm/data/pile/00_0.json \
    --val_path /net/nfs.cirrascale/allennlp/willm/data/pile/val.jsonl \
    --prompts_save_path /net/nfs.cirrascale/allennlp/willm/ngram-copying/data/iid/prompts.jsonl \
    --val_save_path /net/nfs.cirrascale/allennlp/willm/ngram-copying/data/iid/val.jsonl
```

Or to generate models with 100 samples per domain (besides one where there are not 100):
```shell
mkdir /net/nfs.cirrascale/allennlp/willm/ngram-copying/data/by-domain
python sample_prompts_and_val.py \
     --n_samples 100 \
    --by_domain \
    --train_path /net/nfs.cirrascale/allennlp/willm/data/pile/00_0.json \
    --val_path /net/nfs.cirrascale/allennlp/willm/data/pile/val.jsonl \
    --prompts_save_path=$ROOT/data/by-domain/prompts.jsonl \
    --val_save_path=$ROOT/data/by-domain/val.jsonl
```

To generate Dolma Reddit data:
```shell
mkdir /net/nfs.cirrascale/allennlp/willm/ngram-copying/data/dolma
DOLMA=/net/nfs.cirrascale/allennlp/willm/data/v3_small
python sample_prompts_and_val.py \
    --format "dolma" \
    --n_samples 500 \
    --date_cutoff "2019-12-31T11:59:59+00:00" \
    --val_path /net/nfs.cirrascale/allennlp/willm/data/v3_small/dolma_reddit/val/val_reddit_uniform.jsonl \
    --val_save_path /net/nfs.cirrascale/allennlp/willm/ngram-copying/data/dolma/reddit.jsonl
```

To generate Dolma CC data:
```shell
python sample_prompts_and_val.py \
    --format "dolma" \
    --n_samples 500 \
    --date_cutoff "2019-12-31" \
    --trim_date \
    --val_path /net/nfs.cirrascale/allennlp/willm/data/v3_small/dolma_pes2o/val/val_pes2o.jsonl \
    --val_save_path /net/nfs.cirrascale/allennlp/willm/ngram-copying/data/dolma/pes2o.jsonl
```

### Generating Data from Models

To generate by domain (only use Pythia-12b though):
```bash
PLENGTHS="1 10 100" scripts/generate_by_model.sh by-domain by-domain
```

To do the same for deduped (only use Pythia-12b though):
```bash
SUFFIX="-deduped" PLENGTHS="1 10 100" scripts/generate_by_model.sh by-domain by-domain-deduped
```

To generate by model (from IID data):
```bash
scripts/generate_by_model.sh iid by-model
cat $ROOT/lm-generations/by-model/*.jsonl > $ROOT/lm-generations/by-model.jsonl
```

To generate by decoding:
```bash
scripts/generate_by_decoding.sh
cat $ROOT/lm-generations/by-decoding/*.jsonl > $ROOT/lm-generations/by-decoding.jsonl
```

To generate from Pythia-12B deduped:
```bash
scripts/generate_deduped.sh
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

To visualize data once it has been generated, you can export as a CSV and import into Google Drive:

```bash
python write_samples_csv.py \
	$ROOT/lm-generations/by-domain/pythia-12b.jsonl \
	~/by-domain.csv \
	--prompts $ROOT/data/by-domain/prompts.jsonl
```

### Feeding Generated Data Through the API

First start the Rusty DAWG server. Then, to run the full analysis (about 48 hours), you can run:
```bash
scripts/query_cdawgs.sh
```

You can also copy and paste specific commands from that script to just run parts of it.

### To Compute Completion Loss Experiments

To sample more validation data:
```bash
mkdir $ROOT/data/completion-loss
python sample_prompts_and_val.py --n_samples 5000 \
    --prompts_save_path $ROOT/data/completion-loss/prompts.jsonl \
    --val_save_path $ROOT/data/completion-loss/val.jsonl
```

To launch all jobs as Beaker bash jobs (TODO: redo this with larger):

```bash
scripts/compute_losses.sh
```

To run just one:

```bash
printf "pythia-12b-loss" | gantry run \
    --workspace ai2/rusty-dawg \
    --cluster ai2/allennlp-cirrascale \
    --venv base \
    --budget ai2/allennlp \
    --priority normal \
    --gpus 1 -- python completion_loss.py \
    $ROOT/data/completion-loss/val.jsonl \
    $ROOT/results/completion-loss/pythia-12b-BIG.json
```

### To Plot Results

```bash
python plot_novelty.py -n=100 --log-scale --no-plen
```

## Cosmopedia Experiments

Roughly the same workflow as for the Pile. To sample prompts:

```bash
python sample_prompts_and_val.py \
    --train_path /net/nfs.cirrascale/allennlp/willm/data/cosmopedia/cat.jsonl \
    --prompts_save_path /net/nfs.cirrascale/allennlp/willm/ngram-copying/data/cosmopedia/prompts-iid.jsonl \
    --n_samples 500 \
    --format cosmopedia \
    --tokenizer HuggingFaceTB/cosmo-1b \
    --no_val
```

```bash
scripts/generate_cosmopedia.sh
```
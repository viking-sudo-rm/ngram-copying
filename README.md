# N-gram Copying Experiments

## Usage

Prompts can be sampled from the Pile via:
```
python sample_prompts.py
```

Example of how to generate from a specific model:
```
MODEL=pythia-70m-deduped
python generate_from_lm.py \
    EleutherAI/${MODEL} \
    /net/nfs.cirrascale/allennlp/willm/ngram-copying/prompts.jsonl \
    /net/nfs.cirrascale/allennlp/willm/ngram-copying/gen.jsonl \
    --sample
```

Models: 70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b, 12b  ([more information](https://huggingface.co/EleutherAI/pythia-6.9b))

To just run across all models, can do:
```
OUT_DIR=/net/nfs.cirrascale/allennlp/willm/ngram-copying/lm-generations ./generate_from_lms.sh
```
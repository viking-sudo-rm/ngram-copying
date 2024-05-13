#!/bin/bash

ROOT=${ROOT:-"/net/nfs.cirrascale/allennlp/willm/ngram-copying"}
PROMPTS_PATH=$ROOT/data/iid/prompts.jsonl
OUT_DIR=$ROOT/lm-generations
N_TOKENS=1000

echo "========== Generating 'pythia-12b-deduped' =========="
mkdir $OUT_DIR/deduped

printf "pythia-12b-deduped" | gantry run \
    --workspace ai2/rusty-dawg \
    --cluster ai2/allennlp-cirrascale \
    --venv base \
    --budget=ai2/allennlp \
    --gpus=1 -- python generate_from_lm.py \
        EleutherAI/pythia-12b-deduped \
        $PROMPTS_PATH \
        $OUT_DIR/deduped/pythia-12b-deduped.jsonl \
        --n_tokens=$N_TOKENS \
        --prompt_lengths 0 1 10 100 \
        --sample
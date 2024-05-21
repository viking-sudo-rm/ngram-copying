#!/bin/bash

ROOT=${ROOT:-"/net/nfs.cirrascale/allennlp/willm/ngram-copying"}
PROMPTS_PATH=$ROOT/data/cosmopedia/prompts-iid.jsonl
OUT_DIR=$ROOT/lm-generations/cosmopedia
N_TOKENS=1000
PLENGTHS="0 1 10 100"
# FIXME: Unify with decoding script??? Just invoke different arguments?

echo "===== sample ====="
printf "sample" | gantry run \
    --workspace ai2/rusty-dawg \
    --cluster ai2/allennlp-cirrascale \
    --venv base \
    --budget=ai2/allennlp \
    --priority normal \
    --gpus 1 -- python generate_from_lm.py \
        HuggingFaceTB/cosmo-1b \
        $PROMPTS_PATH \
        $OUT_DIR/sample.jsonl \
        --n_tokens $N_TOKENS \
        --prompt_lengths $PLENGTHS \
        --sample

echo "===== topk ====="
printf "topk" | gantry run \
    --workspace ai2/rusty-dawg \
    --cluster ai2/allennlp-cirrascale \
    --venv base \
    --budget=ai2/allennlp \
    --priority normal \
    --gpus 1 -- python generate_from_lm.py \
        HuggingFaceTB/cosmo-1b \
        $PROMPTS_PATH \
        $OUT_DIR/topk.jsonl \
        --n_tokens $N_TOKENS \
        --prompt_lengths $PLENGTHS \
        -k 20 80 160

echo "===== topp ====="
printf "topp" | gantry run \
    --workspace ai2/rusty-dawg \
    --cluster ai2/allennlp-cirrascale \
    --venv base \
    --budget=ai2/allennlp \
    --priority normal \
    --gpus 1 -- python generate_from_lm.py \
        HuggingFaceTB/cosmo-1b \
        $PROMPTS_PATH \
        $OUT_DIR/topp.jsonl \
        --n_tokens $N_TOKENS \
        --prompt_lengths $PLENGTHS \
        -p 0.85 0.90 0.95

echo "===== temp ====="
printf "temp" | gantry run \
    --workspace ai2/rusty-dawg \
    --cluster ai2/allennlp-cirrascale \
    --venv base \
    --budget=ai2/allennlp \
    --priority normal \
    --gpus 1 -- python generate_from_lm.py \
        HuggingFaceTB/cosmo-1b \
        $PROMPTS_PATH \
        $OUT_DIR/temp.jsonl \
        --n_tokens $N_TOKENS \
        --prompt_lengths $PLENGTHS \
        -t 0.50 0.85 0.90 0.95 1.05 1.10

for b in "1 4 8"; do
    echo "===== beam$b ====="
    printf "beam$b" | gantry run \
    --workspace ai2/rusty-dawg \
    --cluster ai2/allennlp-cirrascale \
    --venv base \
    --budget=ai2/allennlp \
    --priority normal \
    --memory 100g \
    --gpus 1 -- python generate_from_lm.py \
        HuggingFaceTB/cosmo-1b \
        $PROMPTS_PATH \
        $OUT_DIR/beam$b.jsonl \
        --n_tokens $N_TOKENS \
        --prompt_lengths $PLENGTHS \
        -p 0.85 0.90 0.95 \
        -b $b \
        --swap_space 64
done
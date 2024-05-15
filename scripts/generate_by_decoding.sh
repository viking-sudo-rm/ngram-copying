#!/bin/bash

ROOT=${ROOT:-"/net/nfs.cirrascale/allennlp/willm/ngram-copying"}
PROMPTS_PATH=$ROOT/data/iid/prompts.jsonl
OUT_DIR=$ROOT/lm-generations
N_TOKENS=1000
PLENGTHS="0 1 10 100"

echo "========== Generating 'by-decoding' =========="
mkdir $OUT_DIR/by-decoding

echo "=== topk ==="
printf "topk" | gantry run \
    --workspace ai2/rusty-dawg \
    --cluster ai2/allennlp-cirrascale \
    --venv base \
    --budget=ai2/allennlp \
    --priority normal \
    --gpus 1 -- python generate_from_lm.py \
        EleutherAI/pythia-12b \
        $PROMPTS_PATH \
        $OUT_DIR/by-decoding/topk.jsonl \
        --n_tokens=$N_TOKENS \
        --prompt_lengths $PLENGTHS \
        -k 20 80 160

echo "=== topp ==="
printf "topp" | gantry run \
    --workspace ai2/rusty-dawg \
    --cluster ai2/allennlp-cirrascale \
    --venv base \
    --budget=ai2/allennlp \
    --priority normal \
    --gpus 1 -- python generate_from_lm.py \
        EleutherAI/pythia-12b \
        $PROMPTS_PATH \
        $OUT_DIR/by-decoding/topp.jsonl \
        --n_tokens=$N_TOKENS \
        --prompt_lengths $PLENGTHS \
        -p 0.85 0.90 0.95

echo "=== temp ==="
printf "temp" | gantry run \
    --workspace ai2/rusty-dawg \
    --cluster ai2/allennlp-cirrascale \
    --venv base \
    --budget=ai2/allennlp \
    --priority normal \
    --gpus 1 -- python generate_from_lm.py \
        EleutherAI/pythia-12b \
        $PROMPTS_PATH \
        $OUT_DIR/by-decoding/temp.jsonl \
        --n_tokens=$N_TOKENS \
        --prompt_lengths $PLENGTHS \
        -t 0.00 0.85 0.90 0.95 1.05 1.10 2.00

echo "=== beam1 ==="
printf "beam1" | gantry run \
    --workspace ai2/rusty-dawg \
    --cluster ai2/allennlp-cirrascale \
    --venv base \
    --budget ai2/allennlp \
    --priority normal \
    --gpus 1 -- python generate_from_lm.py \
        EleutherAI/pythia-12b \
        $PROMPTS_PATH \
        $OUT_DIR/by-decoding/beam1.jsonl \
        --n_tokens=$N_TOKENS \
        --prompt_lengths $PLENGTHS \
        -b 1 

echo "=== beam4 ==="
printf "beam4" | gantry run \
    --workspace ai2/rusty-dawg \
    --cluster ai2/allennlp-cirrascale \
    --venv base \
    --budget ai2/allennlp \
    --priority normal \
    --memory 100g \
    --gpus 1 -- python generate_from_lm.py \
        EleutherAI/pythia-12b \
        $PROMPTS_PATH \
        $OUT_DIR/by-decoding/beam4.jsonl \
        --n_tokens=$N_TOKENS \
        --prompt_lengths $PLENGTHS \
        -b 4 \
        --swap_space 64

echo "=== beam8 ==="
printf "beam8" | gantry run \
    --workspace ai2/rusty-dawg \
    --cluster ai2/allennlp-cirrascale \
    --venv base \
    --budget=ai2/allennlp \
    --priority normal \
    --memory 100g \
    --gpus 1 -- python generate_from_lm.py \
        EleutherAI/pythia-12b \
        $PROMPTS_PATH \
        $OUT_DIR/by-decoding/beam.jsonl \
        --n_tokens=$N_TOKENS \
        --prompt_lengths $PLENGTHS \
        -b 8 \
        --swap_space 64

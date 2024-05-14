#!/bin/bash

ROOT=${ROOT:-"/net/nfs.cirrascale/allennlp/willm/ngram-copying"}
PROMPTS_PATH=$ROOT/data/iid/prompts.jsonl
OUT_DIR=$ROOT/lm-generations
N_TOKENS=1000
PLENGTHS="0 1 10 100"

echo "========== Generating 'by-decoding' =========="
mkdir $OUT_DIR/by-decoding

python generate_from_lm.py \
    EleutherAI/pythia-12b \
    $PROMPTS_PATH \
    $OUT_DIR/by-decoding/topk.jsonl \
    --n_tokens=$N_TOKENS \
    --prompt_lengths $PLENGTHS \
    -k 20 80 160

echo "=== topp ==="
python generate_from_lm.py \
    EleutherAI/pythia-12b \
    $PROMPTS_PATH \
    $OUT_DIR/by-decoding/topp.jsonl \
    --n_tokens=$N_TOKENS \
    --prompt_lengths $PLENGTHS \
    -p 0.85 0.90 0.95

echo "=== temp ==="
python generate_from_lm.py \
    EleutherAI/pythia-12b \
    $PROMPTS_PATH \
    $OUT_DIR/by-decoding/temp.jsonl \
    --n_tokens=$N_TOKENS \
    --prompt_lengths $PLENGTHS \
    -t 0.00 0.85 0.90 0.95 1.05 1.10 2.00

echo "=== beam ==="
python generate_from_lm.py \
    EleutherAI/pythia-12b \
    $PROMPTS_PATH \
    $OUT_DIR/by-decoding/beam.jsonl \
    --n_tokens=$N_TOKENS \
    --prompt_lengths $PLENGTHS \
    -b 1 2 3
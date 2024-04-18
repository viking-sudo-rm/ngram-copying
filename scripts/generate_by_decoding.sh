#!/bin/bash

ROOT=${ROOT:-"/net/nfs.cirrascale/allennlp/willm/ngram-copying"}
PROMPTS_PATH=$ROOT/data/by-domain/prompts.jsonl
OUT_DIR=$ROOT/lm-generations
N_TOKENS=1000

echo "========== Generating 'decoding' =========="
mkdir $OUT_DIR/decoding

echo "=== topk ==="
printf "willm-topk" | gantry run \
    --workspace ai2/rusty-dawg \
    --cluster ai2/allennlp-cirrascale \
    --venv base \
    --budget=ai2/allennlp \
    --gpus=1 -- python generate_from_lm.py \
        EleutherAI/pythia-12b \
        $PROMPTS_PATH \
        $OUT_DIR/decoding/topk.jsonl \
        --n_tokens=$N_TOKENS \
        -k 20 80 160

echo "=== topp ==="
printf "willm-topk" | gantry run \
    --workspace ai2/rusty-dawg \
    --cluster ai2/allennlp-cirrascale \
    --venv base \
    --budget=ai2/allennlp \
    --gpus=1 -- python generate_from_lm.py \
        EleutherAI/pythia-12b \
        $PROMPTS_PATH \
        $OUT_DIR/decoding/topp.jsonl \
        --n_tokens=$N_TOKENS \
        -p 0.85 0.90 0.95

echo "=== temp ==="
printf "willm-temp" | gantry run \
    --workspace ai2/rusty-dawg \
    --cluster ai2/allennlp-cirrascale \
    --venv base \
    --budget=ai2/allennlp \
    --gpus=1 -- python generate_from_lm.py \
        EleutherAI/pythia-12b \
        $PROMPTS_PATH \
        $OUT_DIR/decoding/temp.jsonl \
        --n_tokens=$N_TOKENS \
        -t 0.85 0.90 0.95 1.05 1.10 2.00

echo "=== greedy ==="
printf "willm-greedy" | gantry run \
    --workspace ai2/rusty-dawg \
    --cluster ai2/allennlp-cirrascale \
    --venv base \
    --budget=ai2/allennlp \
    --gpus=1 -- python generate_from_lm.py \
        EleutherAI/pythia-12b \
        $PROMPTS_PATH \
        $OUT_DIR/decoding/greedy.jsonl \
        --n_tokens=$N_TOKENS \
        --greedy

echo "=== beam ==="
printf "willm-beam" | gantry run \
    --workspace ai2/rusty-dawg \
    --cluster ai2/allennlp-cirrascale \
    --venv base \
    --budget=ai2/allennlp \
    --gpus=1 -- python generate_from_lm.py \
        EleutherAI/pythia-12b \
        $PROMPTS_PATH \
        $OUT_DIR/decoding/beam.jsonl \
        --n_tokens=$N_TOKENS \
        -b 2 4 8

# cat $OUT_DIR/decoding/*.jsonl > $OUT_DIR/decoding.jsonl
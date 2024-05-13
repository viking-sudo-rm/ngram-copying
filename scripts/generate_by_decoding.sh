#!/bin/bash

ROOT=${ROOT:-"/net/nfs.cirrascale/allennlp/willm/ngram-copying"}
PROMPTS_PATH=$ROOT/data/iid/prompts.jsonl
OUT_DIR=$ROOT/lm-generations
N_TOKENS=1000

echo "========== Generating 'by-decoding' =========="
mkdir $OUT_DIR/by-decoding

echo "=== topk ==="
printf "topk" | gantry run \
    --workspace ai2/rusty-dawg \
    --cluster ai2/allennlp-cirrascale \
    --venv base \
    --budget=ai2/allennlp \
    --gpus=1 -- python generate_from_lm.py \
        EleutherAI/pythia-12b \
        $PROMPTS_PATH \
        $OUT_DIR/by-decoding/topk.jsonl \
        --n_tokens=$N_TOKENS \
        --prompt_lengths 0 1 10 100 \
        -k 20 80 160

echo "=== topp ==="
printf "topp" | gantry run \
    --workspace ai2/rusty-dawg \
    --cluster ai2/allennlp-cirrascale \
    --venv base \
    --budget=ai2/allennlp \
    --gpus=1 -- python generate_from_lm.py \
        EleutherAI/pythia-12b \
        $PROMPTS_PATH \
        $OUT_DIR/by-decoding/topp.jsonl \
        --n_tokens=$N_TOKENS \
        --prompt_lengths 0 1 10 100 \
        -p 0.85 0.90 0.95

echo "=== temp ==="
printf "temp" | gantry run \
    --workspace ai2/rusty-dawg \
    --cluster ai2/allennlp-cirrascale \
    --venv base \
    --budget=ai2/allennlp \
    --gpus=1 -- python generate_from_lm.py \
        EleutherAI/pythia-12b \
        $PROMPTS_PATH \
        $OUT_DIR/by-decoding/temp.jsonl \
        --n_tokens=$N_TOKENS \
        --prompt_lengths 0 1 10 100 \
        -t 0.00 0.85 0.90 0.95 1.05 1.10 2.00

echo "=== beam ==="
# Gets slow for larger values of b, because whole beam needs to be completed with VLLM.
printf "beam" | gantry run \
    --workspace ai2/rusty-dawg \
    --cluster ai2/allennlp-cirrascale \
    --venv base \
    --budget=ai2/allennlp \
    --gpus=1 -- python generate_from_lm.py \
        EleutherAI/pythia-12b \
        $PROMPTS_PATH \
        $OUT_DIR/by-decoding/beam.jsonl \
        --n_tokens=$N_TOKENS \
        --prompt_lengths 0 1 10 100 \
        -b 1 2 3

### DEBUG ###
# python generate_from_lm.py \
#     EleutherAI/pythia-70m \
#     $ROOT/data/iid/small-prompts.jsonl \
#     $OUT_DIR/by-decoding/beam.jsonl \
#     --n_tokens=$N_TOKENS \
#     --prompt_lengths 0 \
#     --sample
#     # -b 4

# Aggregate all the decoding results.
# cat $OUT_DIR/by-decoding/*.jsonl > $OUT_DIR/by-decoding.jsonl
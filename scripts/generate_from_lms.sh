#!/bin/bash

ROOT=${ROOT:-"/net/nfs.cirrascale/allennlp/willm/ngram-copying"}
PROMPTS_PATH=$ROOT/data/by-domain/prompts.jsonl
OUT_DIR=$ROOT/lm-generations
N_TOKENS=1000

SUFFIX=""
# SUFFIX="-deduped"
DIR=$OUT_DIR/models$SUFFIX

# Fail if already exists.
mkdir $OUT_DIR


echo "========== Generating 'models$SUFFIX' =========="
SIZES=("70m" "160m" "410m" "1b" "1.4b" "2.8b" "6.9b" "12b")
mkdir $DIR
for idx in "${!SIZES[@]}"; do
    model="pythia-${SIZES[idx]}$SUFFIX"
    echo "===== $model ====="
    printf "$model" | gantry run \
        --workspace ai2/rusty-dawg \
        --cluster ai2/allennlp-cirrascale \
        --venv base \
        --budget=ai2/allennlp \
        --gpus=1 -- python generate_from_lm.py \
            EleutherAI/$model \
            $PROMPTS_PATH \
            $DIR/$model.jsonl \
            --n_tokens=$N_TOKENS \
            --sample
done
cat $DIR/*.jsonl > $OUT_DIR/models$SUFFIX.jsonl


# echo "\n"
# echo "========== Generating 'decoding' =========="
# mkdir $OUT_DIR/decoding

# echo "=== topk ==="
# python generate_from_lm.py \
#     EleutherAI/pythia-12b \
#     $PROMPTS_PATH \
#     $OUT_DIR/decoding/topk.jsonl \
#     --n_tokens=$N_TOKENS \
#     --batch_size=10 \
#     -k 20 80 160

# echo "=== topp ==="
# python generate_from_lm.py \
#     EleutherAI/pythia-12b \
#     $PROMPTS_PATH \
#     $OUT_DIR/decoding/topp.jsonl \
#     --n_tokens=$N_TOKENS \
#     --batch_size=10 \
#     -p 0.85 0.90 0.95

# echo "=== temp1 ==="
# python generate_from_lm.py \
#     EleutherAI/pythia-12b \
#     $PROMPTS_PATH \
#     $OUT_DIR/decoding/temp1.jsonl \
#     --n_tokens=$N_TOKENS \
#     --batch_size=10 \
#     -t 0.85 0.90 0.95

# echo "=== temp2 ==="
# python generate_from_lm.py \
#     EleutherAI/pythia-12b \
#     $PROMPTS_PATH \
#     $OUT_DIR/decoding/temp2.jsonl \
#     --n_tokens=$N_TOKENS \
#     --batch_size=10 \
#     -t 1.05 1.10 2.00

# ## STOPPED HERE

# echo "=== greedy ==="
# python generate_from_lm.py \
#     EleutherAI/pythia-12b \
#     $PROMPTS_PATH \
#     $OUT_DIR/decoding/greedy \
#     --n_tokens=$N_TOKENS \
#     --batch_size=10 \
#     --greedy

# echo "=== max n ==="
# python generate_from_lm.py \
#     EleutherAI/pythia-12b \
#     $PROMPTS_PATH \
#     $OUT_DIR/decoding/greedy.jsonl \
#     --n_tokens=$N_TOKENS \
#     --batch_size=10 \
#     -r 2 3 4

# echo "=== beam ==="
# python generate_from_lm.py \
#     EleutherAI/pythia-12b \
#     $PROMPTS_PATH \
#     $OUT_DIR/decoding/beam.jsonl \
#     --n_tokens=$N_TOKENS \
#     --batch_size=1 \  # Beam search uses more memory!
#     -b 2 4 8

# cat $OUT_DIR/decoding/*.jsonl > $OUT_DIR/decoding.jsonl
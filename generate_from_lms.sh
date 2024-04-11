#!/usr/bin/bash

ROOT=${ROOT:-"/net/nfs.cirrascale/allennlp/willm/ngram-copying"}
PROMPTS_PATH=$ROOT/prompts.jsonl
OUT_DIR=$ROOT/lm-generations
N_TOKENS=1000

# Fail if already exists.
mkdir $OUT_DIR


echo "========== Generating 'models' =========="
SIZES=("70m" "160m" "410m" "1b" "1.4b" "2.8b" "6.9b" "12b")
BATCH_SIZES=("50" "50" "50" "50" "50" "50" "20" "10")
mkdir $OUT_DIR/models
for idx in "${!SIZES[@]}"; do
    size=${SIZES[idx]}
    batch_size=${BATCH_SIZES[idx]}
    MODEL="pythia-$size"
    echo "===== ${MODEL} ====="
    python generate_from_lm.py \
        EleutherAI/${MODEL} \
        $PROMPTS_PATH \
        $OUT_DIR/models/${MODEL}.jsonl \
        --n_tokens=$N_TOKENS \
        --batch_size=$batch_size \
        --sample
done
cat $OUT_DIR/models/*.jsonl > $OUT_DIR/models.jsonl


echo "\n"
echo "========== Generating 'models-deduped' =========="
SIZES=("70m" "160m" "410m" "1b" "1.4b" "2.8b" "6.9b" "12b")
BATCH_SIZES=("50" "50" "50" "50" "50" "50" "20" "10")
mkdir $OUT_DIR/models-deduped
for idx in "${!SIZES[@]}"; do
    size=${SIZES[idx]}
    batch_size=${BATCH_SIZES[idx]}
    MODEL="pythia-$size-deduped"
    echo "===== ${MODEL} ====="
    python generate_from_lm.py \
        EleutherAI/${MODEL} \
        $PROMPTS_PATH \
        $OUT_DIR/models-deduped/${MODEL}.jsonl \
        --n_tokens=$N_TOKENS \
        --batch_size=$batch_size \
        --sample
done
cat $OUT_DIR/models-deduped/*.jsonl > $OUT_DIR/models-deduped.jsonl


echo "\n"
echo "========== Generating 'decoding' =========="
mkdir $OUT_DIR/decoding
echo "=== sample #1 ==="
python generate_from_lm.py \
    EleutherAI/pythia-12b \
    $PROMPTS_PATH \
    $OUT_DIR/decoding/sample1 \
    --n_tokens=$N_TOKENS \
    --batch_size=10 \
    --sample \
    -k 20 80 160
echo "=== sample #2 ==="
python generate_from_lm.py \
    EleutherAI/pythia-12b \
    $PROMPTS_PATH \
    $OUT_DIR/decoding/sample2 \
    --n_tokens=$N_TOKENS \
    --batch_size=10 \
    -p 0.85 0.90 0.95 \
    -t 0.85 0.90 0.95 1.05 2.00
echo "=== greedy #1 ==="
python generate_from_lm.py \
    EleutherAI/pythia-12b \
    $PROMPTS_PATH \
    $OUT_DIR/decoding/greedy \
    --n_tokens=$N_TOKENS \
    --batch_size=10 \
    --greedy \
    -r 2 3 4
echo "=== sample #2 ==="
python generate_from_lm.py \
    EleutherAI/pythia-12b \
    $PROMPTS_PATH \
    $OUT_DIR/decoding/beam \
    --n_tokens=$N_TOKENS \
    --batch_size=1 \  # Beam search uses more memory!
    -b 2 4 8
cat $OUT_DIR/decoding/*.jsonl > $OUT_DIR/decoding.jsonl
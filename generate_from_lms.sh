#!/usr/bin/bash

N_TOKENS=1000
PROMPTS_PATH=/net/nfs.cirrascale/allennlp/willm/ngram-copying/prompts.jsonl

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
echo "========== Generating 'dedup-models' =========="
mkdir $OUT_DIR/dedup-models
for idx in "${!SIZES[@]}"; do
    size=${SIZES[idx]}
    batch_size=${BATCH_SIZES[idx]}
    MODEL="pythia-$size-deduped"
    echo "===== ${MODEL} ====="
    python generate_from_lm.py \
        EleutherAI/${MODEL} \
        $PROMPTS_PATH \
        $OUT_DIR/dedup-models/${MODEL}.jsonl \
        --n_tokens=$N_TOKENS \
        --batch_size=$batch_size \
        --sample
done
cat $OUT_DIR/dedup-models/*.jsonl > $OUT_DIR/dedup-models.jsonl


echo "\n"
echo "========== Generating 'decoding' =========="
mkdir $OUT_DIR/decoding
MODELS=("pythia-12b")
for MODEL in "${MODELS[@]}"; do
    echo "===== ${MODEL} ====="
    python generate_from_lm.py \
        EleutherAI/${MODEL} \
        $PROMPTS_PATH \
        $OUT_DIR/decoding/${MODEL}.jsonl \
        --n_tokens=$N_TOKENS \
        --batch_size=10 \
        --sample \
        -k 20 80 160 \
        -p 0.85 0.90 0.95 \
        -t 0.85 0.90 0.95 1.05 2.00 \
        --greedy \
        -b 2 4 8 \
        -r 2 3 4
done
cat $OUT_DIR/decoding/*.jsonl > $OUT_DIR/decoding.jsonl

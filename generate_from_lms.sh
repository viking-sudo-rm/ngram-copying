#!/usr/bin/bash

N_TOKENS=1000
PROMPTS_PATH=/net/nfs.cirrascale/allennlp/willm/ngram-copying/prompts.jsonl

# Fail if already exists.
mkdir $OUT_DIR

echo "========== Generating data from different models =========="
SIZES=("70m" "160m" "410m" "1b" "1.4b" "2.8b" "6.9b" "12b")
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
        --sample
    MODEL="pythia-$size-deduped"
    echo "===== ${MODEL} ====="
    python generate_from_lm.py \
        EleutherAI/${MODEL} \
        $PROMPTS_PATH \
        $OUT_DIR/models/${MODEL}.jsonl \
        --n_tokens=$N_TOKENS \
        --sample
done

cat $OUT_DIR/models/*.jsonl > $OUT_DIR/models.jsonl


echo "\n"
echo "========== Generating decoding data =========="
mkdir $OUT_DIR/decoding

MODELS=("pythia-12b" "pythia-12b-deduped")
for MODEL in "${MODELS[@]}"; do
    echo "===== ${MODEL} ====="
    python generate_from_lm.py \
        EleutherAI/${MODEL} \
        $PROMPTS_PATH \
        $OUT_DIR/decoding/${MODEL}.jsonl \
        --n_tokens=$N_TOKENS \
        -k 20 40 80 160 320 \
        -p 0.85 0.90 0.95 1.00 \
        -t 0.85 0.90 0.95 1.00 1.05 2.00 \
        -b 1 2 4 6
done

cat $OUT_DIR/decoding/*.jsonl > $OUT_DIR/decoding.jsonl

#!/bin/bash

ROOT=${ROOT:-"/net/nfs.cirrascale/allennlp/willm/ngram-copying"}
PROMPTS_PATH=$ROOT/prompts.jsonl
OUT_DIR=$ROOT/lm-generations
N_TOKENS=1000

SUFFIX=""
# SUFFIX="-deduped"

# Fail if already exists.
mkdir $OUT_DIR


echo "========== Generating 'models$suffix' =========="
SIZES=("70m" "160m" "410m" "1b" "1.4b" "2.8b" "6.9b" "12b")
mkdir $OUT_DIR/models$suffix
for idx in "${!SIZES[@]}"; do
    size=${SIZES[idx]}
    MODEL="pythia-$size$suffix"
    echo "===== ${MODEL} ====="
    python generate_from_lm.py \
        EleutherAI/${MODEL} \
        $PROMPTS_PATH \
        $OUT_DIR/models$suffix/${MODEL}.jsonl \
        --n_tokens=$N_TOKENS \
        --sample
done
cat $OUT_DIR/models$suffix/*.jsonl > $OUT_DIR/models$suffix.jsonl
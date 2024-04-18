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
    printf "\n" | gantry run \
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
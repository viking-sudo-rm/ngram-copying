#!/bin/bash

ROOT=${ROOT:-"/net/nfs.cirrascale/allennlp/willm/ngram-copying"}
DATA=${1:-"iid"}
SAVE=${2:-"by-model"}
PROMPTS_PATH=$ROOT/data/$DATA/prompts.jsonl
OUT_DIR=$ROOT/lm-generations
N_TOKENS=1000
SUFFIX=""

# Fail if already exists.
mkdir $OUT_DIR


echo "========== Generating '$SAVE' =========="
SIZES=("70m" "160m" "410m" "1b" "1.4b" "2.8b" "6.9b" "12b")
mkdir $OUT_DIR/$SAVE
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
            $OUT_DIR/$SAVE/$model.jsonl \
            --n_tokens=$N_TOKENS \
            --sample
done
# cat $DIR/*.jsonl > $OUT_DIR/models$SUFFIX.jsonl
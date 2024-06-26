#!/bin/bash

ROOT=${ROOT:-"/net/nfs.cirrascale/allennlp/willm/ngram-copying"}

SIZES=("70m" "160m" "410m" "1b" "1.4b" "2.8b" "6.9b" "12b")
SAMPLE="iid"  # TODO: Redo with "completion-loss"

mkdir $ROOT/results/completion-loss
for idx in "${!SIZES[@]}"; do
    model="pythia-${SIZES[idx]}"
    echo "===== $model-loss ====="
    printf "$model-loss" | gantry run \
        --workspace ai2/rusty-dawg \
        --cluster ai2/allennlp-cirrascale \
        --venv base \
        --budget ai2/allennlp \
        --priority normal \
        --gpus 1 -- python completion_loss.py \
            $ROOT/data/$SAMPLE/val.jsonl \
            $ROOT/results/completion-loss/$model.json \
            --model EleutherAI/$model
done
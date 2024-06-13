#!/bin/bash
# Sending one document at a time, this seems to take ~10sec/doc, or 100 tok/sec
# Probably a lot of this is network overhead and bottleneck (maybe batching would help?)
# Normally, the DAWG seems to be able to run on disk comfortably at 1000 tok/sec
# --return-next-tokens "-1" is slow!!!

ROOT=${ROOT:-"/net/nfs.cirrascale/allennlp/willm/ngram-copying"}
BATCH_SIZE=100
TIMEOUT=999999999999

# DEBUG
python query_cdawgs.py \
    $ROOT/data/debug.jsonl \
    $ROOT/results/debug.json \
    --batch-size $BATCH_SIZE \
    --read-timeout $TIMEOUT \
    --return-entropies

# Pass by-domain val through the CDAWGs.
# python query_cdawgs.py \
#     $ROOT/data/by-domain/val.jsonl \
#     $ROOT/results/val-domain.json \
#     --batch-size $BATCH_SIZE \
#     --read-timeout $TIMEOUT

# Pass iid val through the CDAWGs.
# python query_cdawgs.py \
#     $ROOT/data/iid/val.jsonl \
#     $ROOT/results/val-iid.json \
#     --batch-size $BATCH_SIZE \
#     --read-timeout $TIMEOUT

# Pass completion-loss val through the CDAWGs.
python query_cdawgs.py \
    $ROOT/data/completion-loss/val.jsonl \
    $ROOT/results/val-cl-entropy.json \
    --batch-size $BATCH_SIZE \
    --read-timeout $TIMEOUT \
    --return-entropies

# queries
python query_cdawgs.py \
    $ROOT/data/queries.jsonl \
    $ROOT/results/queries.json \
    --batch-size $BATCH_SIZE \
    --read-timeout $TIMEOUT \
    --text

# echo "=== Generating by-domain results ==="
# python query_cdawgs.py \
#     $ROOT/lm-generations/by-domain/pythia-12b.jsonl \
#     $ROOT/results/by-domain.json \
#     --batch-size $BATCH_SIZE \
#     --read-timeout $TIMEOUT

# echo "=== Generating by-domain-deduped results ==="
# python query_cdawgs.py \
#     $ROOT/lm-generations/deduped/by-domain/pythia-12b-deduped.jsonl \
#     $ROOT/results/by-domain-deduped.json \
#     --batch-size $BATCH_SIZE \
#     --read-timeout $TIMEOUT

# echo "=== Generating by-model results ==="
# python query_cdawgs.py \
#     $ROOT/lm-generations/by-model.jsonl \
#     $ROOT/results/by-model.json \
#     --batch-size $BATCH_SIZE \
#     --read-timeout $TIMEOUT

# echo "=== Generating pythia-12b-deduped results ==="
# python query_cdawgs.py \
#     $ROOT/lm-generations/deduped/pythia-12b-deduped.jsonl \
#     $ROOT/results/deduped/pythia-12b-deduped.json \
#     --batch-size $BATCH_SIZE \
#     --read-timeout $TIMEOUT

# Was crashing a long way into this, so run this way with intermediate cacheing
# mkdir $ROOT/results/by-decoding
# # decs="topp topk temp beam1 beam4 beam8"
# decs="temp"
# for dec in $decs; do
#     echo "=== Generating by-decoding/$dec results ==="
#     python query_cdawgs.py \
#         $ROOT/lm-generations/by-decoding/$dec.jsonl \
#         $ROOT/results/by-decoding/$dec.json \
#         --batch-size $BATCH_SIZE \
#         --read-timeout $TIMEOUT
# done
#!/bin/bash
# Sending one document at a time, this seems to take ~10sec/doc, or 100 tok/sec
# Probably a lot of this is network overhead and bottleneck (maybe batching would help?)
# Normally, the DAWG seems to be able to run on disk comfortably at 1000 tok/sec

ROOT=${ROOT:-"/net/nfs.cirrascale/allennlp/willm/ngram-copying"}
BATCH_SIZE=100
TIMEOUT=9999999999

# Pass by-domain val through the CDAWGs.
python query_cdawgs.py \
    $ROOT/data/by-domain/val.jsonl \
    $ROOT/results/val-domain.json \
    --batch_size $BATCH_SIZE \
    --read_timeout $TIMEOUT

# TODO: mv to by-domain folder and iid folder 
# Get by-domain results on Pythia-12b.
python query_cdawgs.py \
    $ROOT/lm-generations/by-domain/pythia-12b.jsonl \
    $ROOT/results/by-domain.json \
    --batch_size $BATCH_SIZE \
    --read_timeout $TIMEOUT

# Deduped version of by-domain.
python query_cdawgs.py \
    $ROOT/lm-generations/by-domain/pythia-12b.jsonl \
    $ROOT/results/by-domain-deduped.json \
    --batch_size $BATCH_SIZE \
    --read_timeout $TIMEOUT

# Pass iid val through the CDAWGs.
python query_cdawgs.py \
    $ROOT/data/iid/val.jsonl \
    $ROOT/results/val-iid.json \
    --batch_size $BATCH_SIZE \
    --read_timeout $TIMEOUT

# Pass aggregated by-model data through CDAWGs.
python query_cdawgs.py \
    $ROOT/lm-generations/by-model.jsonl \
    $ROOT/results/by-model.json \
    --batch_size $BATCH_SIZE \
    --read_timeout $TIMEOUT

# Pass decoding data through the CDAWGs.
python query_cdawgs.py \
    $ROOT/lm-generations/by-decoding/pythia-12b.jsonl \
    $ROOT/results/by-decoding.json \
    --batch_size $BATCH_SIZE \
    --read_timeout $TIMEOUT
#!/bin/bash
# Sending one document at a time, this seems to take ~10sec/doc, or 100 tok/sec
# Probably a lot of this is network overhead and bottleneck (maybe batching would help?)
# Normally, the DAWG seems to be able to run on disk comfortably at 1000 tok/sec

# Pass 'val' through the CDAWGs.
python query_cdawgs.py \
    /net/nfs.cirrascale/allennlp/willm/ngram-copying/validation/val-20-tokens.jsonl \
    /net/nfs.cirrascale/allennlp/willm/ngram-copying/results/val-20.json \
    --batch_size 5 \
    --read_timeout 200

# Pass 'pythia-12b' through the CDAWGs.
python query_cdawgs.py \
    /net/nfs.cirrascale/allennlp/willm/ngram-copying/lm-generations/models/pythia-12b.jsonl \
    /net/nfs.cirrascale/allennlp/willm/ngram-copying/results/models/pythia-12b.json \
    --batch_size 5 \
    --read_timeout 200

# Pass 'models' through the CDAWGs.
python query_cdawgs.py \
    /net/nfs.cirrascale/allennlp/willm/ngram-copying/lm-generations/models.jsonl \
    /net/nfs.cirrascale/allennlp/willm/ngram-copying/results/models.json \
    --batch_size 5 \
    --read_timeout 200

# Pass 'dedup-models' through the CDAWGs.
python query_cdawgs.py \
    /net/nfs.cirrascale/allennlp/willm/ngram-copying/lm-generations/dedup-models.jsonl \
    /net/nfs.cirrascale/allennlp/willm/ngram-copying/results/dedup-models.json \
    --batch_size 5 \
    --read_timeout 200

# Pass 'decoding' through the CDAWGs.
python query_cdawgs.py \
    /net/nfs.cirrascale/allennlp/willm/ngram-copying/lm-generations/decoding.jsonl \
    /net/nfs.cirrascale/allennlp/willm/ngram-copying/results/decoding.json \
    --batch_size 5 \
    --read_timeout 200
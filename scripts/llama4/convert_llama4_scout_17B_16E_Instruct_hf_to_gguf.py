#!/bin/bash

CONVERTER=./third-party/llama.cpp/convert_hf_to_gguf.py 
HF_SNAPSHOT_DIR=~/.cache/huggingface/models--meta-llama--Llama-4-Scout-17B-16E-Instruct/snapshots/92f3b1597a195b523d8d9e5700e57e4fbb8f20d3/
GGUF_FILE=llama-4-scout-17b-16e-instruct.gguf
OUT_DTYPE=bf16

$CONVERTER $HF_SNAPSHOT_DIR --outfile $GGUF_FILE --outtype $OUT_DTYPE

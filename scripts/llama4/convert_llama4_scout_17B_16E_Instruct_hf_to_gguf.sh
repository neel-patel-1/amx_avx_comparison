#!/bin/bash
CONVERTER_DIR=./third-party/llama.cpp/
CONVERTER=$CONVERTER_DIR//convert_hf_to_gguf.py 
HF_SNAPSHOT_DIR=~/.cache/huggingface/hub/models--meta-llama--Llama-4-Scout-17B-16E-Instruct/snapshots/92f3b1597a195b523d8d9e5700e57e4fbb8f20d3/
GGUF_FILE=llama-4-scout-17b-16e-instruct.gguf
OUT_DTYPE=bf16

python $CONVERTER $HF_SNAPSHOT_DIR --outfile $GGUF_FILE --outtype $OUT_DTYPE

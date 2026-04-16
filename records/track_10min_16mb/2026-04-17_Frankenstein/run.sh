#!/bin/bash

for seed in 42 1337 0; do

	echo "--- SEED $seed --- ";

	SEED=$seed RUN_ID="frankenstein_$seed" \
	SMEAR_GATE=1 SMEAR_GATE_WIDTH=12 \
	GATE_ATTN_OUT=1 GATE_ATTN_SRC=proj GATE_WIDTH=12 \
	QK_GAIN_INIT=5.25 \
	TTT_ENABLED=1 \
	torchrun --standalone --nproc_per_node=8 train_gpt.py

done

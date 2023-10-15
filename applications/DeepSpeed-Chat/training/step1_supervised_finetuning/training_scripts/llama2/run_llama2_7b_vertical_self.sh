#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_step1_long_llama2_7b_CA_v4_vertical_10ct_all_layers_self_attn_norm_v2
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT
#local/jsonfile
#huggyllama/llama-7b
#daryl149/llama-2-7b-chat-hf
#--hostfile="training_scripts/llama2/host_file" --exclude="localhost:0"
#conceptofmind/LLongMA-2-7b-16k
#daryl149/llama-2-7b-hf
deepspeed  main_vertical_self.py \
   --data_path local/jsonfile \
   --data_split 9,1,1 \
   --model_name_or_path daryl149/llama-2-7b-hf \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --max_seq_len 256 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 4  \
   --gradient_accumulation_steps 8 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --offload \
   --deepspeed \
   --print_loss \
   --attn_norm \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log

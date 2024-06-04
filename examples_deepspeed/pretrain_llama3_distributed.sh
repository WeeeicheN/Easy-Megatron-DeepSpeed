#!/bin/bash
# This example script is originally contributed by external user https://github.com/nrailgun
set -ex

workdir=$PWD
rootdir="$workdir""/../.."

######################################
# Path
BASE_PATH="$rootdir""/checkpoints/TestModel"
DS_CONFIG=${BASE_PATH}/deepspeed.json
DATASET_1="$rootdir""/data/test_data/data"
DATASET="1 ${DATASET_1}"
CHECKPOINT_PATH="$rootdir""/checkpoints/TestModel"
TOKENIZER_PATH="$rootdir""/..""/pretrained_model/Meta-Llama-3-8B-Instruct"

######################################
# Device Configs
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6022

num_gpus=8 #$(($(ds_ssh nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)-2))
num_gpus_pernode=8 #$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
num_node=1 #$(( ${num_gpus} / ${num_gpus_pernode} ))

######################################
# Model Configs
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=14336
NUM_LAYERS=32
NUM_HEADS=32
SEQ_LENGTH=8192
NUM_KV_HEADS=8

######################################
# Training Configs
## Parallelism
TP=2
PP=2
MP=1
no_pp="false"
ZERO_STAGE=1
### Data parallel size.
dp_size=$(( ${num_gpus} / ${pp_size} / ${mp_size} ))
### Make sure that micro_batch_size <= global_batch_size*pp_size*mp_size/num_gpus
GLOBAL_BATCH_SIZE=8
MICRO_BATCH_SIZE=1

## Duration
### The main termination condition
train_tokens_in_billion=0.0001
train_tokens=$((${train_tokens_in_billion} * 1000000000))

### train_samples is another termination condition and also affect the number of 
### data samples to be indexed. Since we want to reach the train_tokens
### above, and data efficiency techniques may change num tokens in some samples,
### so we just set this config large enough to make sure we have enough
### processed data and don't terminate by train_samples.
train_samples=$(( 300 * 1000000000 * 2 / ${seq_len} ))

### Another wall-clock time termination condition in minutes. Set it large
### enough to avoid undesired early termination.
exit_duration=30000000

## LR
LR=3e-4
MIN_LR=3e-5
GRAD_CLIP=1

### lr warmup and decay duration
LR_WARMUP_STEPS=1
WEIGHT_DECAY=0.1

lr_warmup_tokens_in_million=0.001
lr_warmup_tokens=$((${lr_warmup_tokens_in_million} * 1000000))
### Here we changed the LR decay tokens to align with total train tokens, since
### related works (e.g., https://arxiv.org/abs/2203.15556) find that setting the
### learning rate schedule to match the number of training tokens results in the
### best final model quality 
lr_decay_tokens_in_billion=${train_tokens_in_billion}
lr_decay_tokens=$((${lr_decay_tokens_in_billion} * 1000000000))
lr_decay_style="cosine"


## Misc
log_interval=1
eval_iters=1
eval_interval=5
### num_save controls how frequent to save checkpoint. num_save=20 means that a
### checkpoint will be saved every 5% of training. For longer training you would
### want larger num_save to save more frequently, and vice versa.
num_save=1
estimated_train_iter=$((${train_tokens} / ${seq_len} / ${global_batch_size}))
### save_interval=$((${estimated_train_iter} / ${num_save}))
save_interval=5

### Activation checkpointing saves GPU memory, but reduces training speed
#activation_checkpoint="true"
activation_checkpoint="false"

### Whether or not log optimizer states (norms, max abs values) to tensorboard.
### This is not required for training and might save GPU memory when turned off.
log_optimizer_state="false"

######################################
# Data Configs
data_options=" \
    --data-path ${data_path} \
    --data-impl mmap"

# Below configuration required for llama model as per llama paper
# --no-query-key-layer-scaling \
# --attention-dropout 0 \
# --hidden-dropout 0 \
# --use-rotary-position-embeddings \
# --untie-embeddings-and-output-weights \
# --swiglu \
# --normalization rmsnorm \
# --disable-bias-linear \
###############################################################################
megatron_options=" \
    --override-opt_param-scheduler \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --tensor-model-parallel-size ${mp_size} \
    --init-method-std ${init_std} \
    --lr-decay-tokens ${lr_decay_tokens} \
    --lr-warmup-tokens ${lr_warmup_tokens} \
    --micro-batch-size ${batch_size} \
    --exit-duration-in-mins ${exit_duration} \
    --global-batch-size ${global_batch_size} \
    --num-layers ${num_layers} \
    --hidden-size ${hidden_size} \
    --num-attention-heads ${num_attn_heads} \
    --seq-length ${seq_len} \
    --max-position-embeddings ${seq_len} \
    --train-tokens ${train_tokens} \
    --train-samples ${train_samples} \
    --lr ${lr} \
    --min-lr ${min_lr} \
    --lr-decay-style ${lr_decay_style} \
    --split 949,50,1 \
    --log-interval ${log_interval} \
    --eval-interval ${eval_interval} \
    --eval-iters ${eval_iters} \
    --save-interval ${save_interval} \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --hysteresis 2 \
    --num-workers ${num_workers} \
    --fp16 \
    --seed ${seed} \
    --load ${checkpoint_path} \
    --save ${checkpoint_path} \
    --no-async-tensor-model-parallel-allreduce \
    --tensorboard-queue-size 1 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --tensorboard-dir ${tensorboard_path}"

if [ "${activation_checkpoint}" = "true" ]; then
megatron_options="${megatron_options} \
    --checkpoint-activations"
fi

if [ "${log_optimizer_state}" = "true" ]; then
megatron_options="${megatron_options} \
    --log-optimizer-states-to-tensorboard"
fi

cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "steps_per_print": 1,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "bf16": {
    "enabled": true
  }
}
EOT

deepspeed_options=""
deepspeed_options=" --deepspeed ${deepspeed_options}"
deepspeed_options=" --deepspeed_config=$DS_CONFIG ${deepspeed_options}"
deepspeed_options=" --zero-stage=$ZERO_STAGE ${deepspeed_options}"
deepspeed_options=" --pipeline-model-parallel-size=$PP ${deepspeed_options}"

if [ "${activation_checkpoint}" = "true" ]; then
  deepspeed_options="--deepspeed-activation-checkpointing ${deepspeed_options}"

  ## old argument for recomputing the transformer layer
  # deepspeed_options="--checkpoint-activations ${deepspeed_options}"

  ## new argument for recomputing the transformer layer
  deepspeed_options="--recompute-granularity full --recompute-method uniform ${deepspeed_options}"
  ## new argument for recomputing only the attention layer
  # deepspeed_options="--recompute-granularity selective ${deepspeed_options}"
fi

if [[ "${no_pp}" = "true" ]]; then
deepspeed_options="${deepspeed_options} \
    --no-pipeline-parallel"
fi

## When saving checkpoint to a storage with cache, their could be consistency
## issue of the pointer to latest checkpoint. Here we find the correct pointer
## and broadcast it to all nodes.
iteration_file="$checkpoint_path/latest_checkpointed_iteration.txt"
iteration_file_2="$checkpoint_path/latest"
iteration=0
for (( node = 0; node <= num_node-1; node++ ))
do
    if $(ssh -q worker-"$node" "test -f \"$iteration_file\""); then
        local_iteration=$(ssh -q worker-"$node" cat $iteration_file)
        iteration=$(( ${local_iteration} > ${iteration} ? ${local_iteration} :  ${iteration} ))
    fi
done
if [[ $iteration -gt 0 ]]; then
    iteration_2="global_step${iteration}"
    ds_ssh "echo $iteration > $iteration_file"
    ds_ssh "echo $iteration_2 > $iteration_file_2"
fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

torchrun $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --tensor-model-parallel-size $TP \
       --pipeline-model-parallel-size $PP \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --ffn-hidden-size $FFN_HIDDEN_SIZE \
       --num-attention-heads $NUM_HEADS \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --seq-length $SEQ_LENGTH \
       --max-position-embeddings $SEQ_LENGTH \
       --train-iters $TRAIN_STEPS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATASET \
       --data-impl mmap \
       --tokenizer-type GPTSentencePieceTokenizer \
       --tokenizer-model $TOKENIZER_PATH \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr $LR \
       --lr-decay-style cosine \
       --min-lr $MIN_LR \
       --weight-decay $WEIGHT_DECAY \
       --clip-grad $GRAD_CLIP \
       --lr-warmup-iters $LR_WARMUP_STEPS \
       --optimizer adam \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --bf16 \
       --no-query-key-layer-scaling \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --use-rotary-position-embeddings \
       --untie-embeddings-and-output-weights \
       --swiglu \
       --normalization rmsnorm \
       --disable-bias-linear \
       --num-key-value-heads $NUM_KV_HEADS \
       $deepspeed_options

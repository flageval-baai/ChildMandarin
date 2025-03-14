#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0,1,2,3" 
# The NCCL_SOCKET_IFNAME variable specifies which IP interface to use for nccl
# communication. More details can be found in
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
# export NCCL_SOCKET_IFNAME=ens4f1
export NCCL_DEBUG=INFO
stage=1 # start from 0 if you need to start from data preparation
stop_stage=2

# The num of machines(nodes) for multi-machine training, 1 is for one machine.
# NFS is required if num_nodes > 1.
num_nodes=1

# The rank of each node or machine, which ranges from 0 to `num_nodes - 1`.
# You should set the node_rank=0 on the first machine, set the node_rank=1
# on the second machine, and so on.
node_rank=0
# The aishell dataset location, please change this to your own path
# make sure of using absolute path. DO-NOT-USE relatvie path!

nj=16
dict=exp/child_conformer_finetune_based_wenetspeech/units.txt #./child_exp/child_conformer_finetune_based_wenetspeech/units.txt

# data_type can be `raw` or `shard`. Typically, raw is used for small dataset,
# `shard` is used for large dataset which is over 1k hours, and `shard` is
# faster on reading data and training.
data_type=raw
num_utts_per_shard=1000

train_set=train
train_config=finetune_wenetspeech_conformer.yaml
dir=exp/child_conformer_finetune_based_wenetspeech

checkpoint= 
num_workers=8
prefetch=500

# use average_checkpoint will get better result
average_checkpoint=True
# decode_checkpoint=$dir/final.pt
average_num=10
decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring" 


deepspeed=false
deepspeed_config=conf/ds_stage2.json
deepspeed_save_states="model_only"

. tools/parse_options.sh || exit 1;


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  mkdir -p $dir
  # You have to rm `INIT_FILE` manually when you resume or restart a
  # multi-machine training.
  INIT_FILE=$dir/ddp_init
  rm -f ${INIT_FILE}  # remove previous INIT_FILE
  init_method=file://$(readlink -f $INIT_FILE)
  echo "$0: init method is $init_method"
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="gloo"
  world_size=`expr $num_gpus \* $num_nodes`
  echo "total gpus is: $world_size"
  cmvn_opts=
  $cmvn && cp data/${train_set}/global_cmvn $dir
  $cmvn && cmvn_opts="--cmvn ${dir}/global_cmvn"

  # train.py rewrite $train_config to $dir/train.yaml with model input
  # and output dimension, and $dir/train.yaml will be used for inference
  # and export.
  if [ ${deepspeed} == true ]; then
    echo "using deepspeed"
    # NOTE(xcsong): deepspeed fails with gloo, see
    #   https://github.com/microsoft/DeepSpeed/issues/2818
    dist_backend="nccl"
    [ ! -f data/$train_set/data.list.filter ] && \
      python tools/filter_uneven_data.py data/$train_set/data.list \
        $data_type $num_gpus $num_utts_per_shard data/$train_set/data.list.filter
    deepspeed --include localhost:$CUDA_VISIBLE_DEVICES \
      wenet/bin/train.py \
        --deepspeed \
        --deepspeed_config ${deepspeed_config} \
        --deepspeed.save_states ${deepspeed_save_states} \
        --ddp.dist_backend $dist_backend \
        --ddp.init_method $init_method \
        --data_type  $data_type \
        --config $train_config \
        --symbol_table  data/dict/lang_char.txt \
        --train_data data/$train_set/data.list.filter \
        --cv_data data/dev/data.list \
        ${checkpoint:+--checkpoint $checkpoint} \
        --model_dir $dir \
        --num_workers ${num_workers} \
        --prefetch ${prefetch} \
        $cmvn_opts \
        --pin_memory
  else
    echo "using torch ddp"
    for ((i = 0; i < $num_gpus; ++i)); do
    {
      gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
      # Rank of each gpu/process used for knowing whether it is
      # the master of a worker.
      rank=`expr $node_rank \* $num_gpus + $i`
      python wenet/bin/train.py --gpu $gpu_id \
        --config $train_config \
        --data_type $data_type \
        --symbol_table $dict \
        --train_data data/$train_set/data.list \
        --cv_data data/dev/data.list \
        ${checkpoint:+--checkpoint $checkpoint} \
        --model_dir $dir \
        --ddp.init_method $init_method \
        --ddp.world_size $world_size \
        --ddp.rank $rank \
        --ddp.dist_backend $dist_backend \
        --num_workers ${num_workers} \
        --prefetch ${prefetch} \
        $cmvn_opts \
        --pin_memory
    } &
    done
    wait
  fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # Test model, please specify the model you want to test by --checkpoint
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg_${average_num}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path $dir  \
      --num ${average_num} \
      --val_best
  fi
  # Please specify decoding_chunk_size for unified streaming and
  # non-streaming model. The default value is -1, which is full chunk
  # for non-streaming inference.
  decoding_chunk_size=
  ctc_weight=0.3
  reverse_weight=0.0 # 0.5
  for mode in ${decode_modes}; do
  {
    test_dir=$dir/test_${mode}
    mkdir -p $test_dir
    python wenet/bin/recognize.py --gpu 0 \
      --mode $mode \
      --config $dir/train.yaml \
      --data_type $data_type \
      --test_data data/test/data.list \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size 1 \
      --penalty 0.0 \
      --dict $dict \
      --ctc_weight $ctc_weight \
      --reverse_weight $reverse_weight \
      --result_file $test_dir/text \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
    python tools/compute-wer.py --char=1 --v=1 \
      data/test/text $test_dir/text > $test_dir/wer
  } &
  done
  wait
fi

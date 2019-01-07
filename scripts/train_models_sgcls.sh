#!/usr/bin/env bash

#export CUDA_LAUNCH_BLOCKING=1

function usage(){
    echo "This is a script that will train all of the models for scene graph classification and then evaluate them"
    echo -e "run: ./script/train_model_sgcls [1|2|3|4|5]"
}
export CUDA_VISIBLE_DEVICES=4
export NUM_GPU=1
export BATCH_SIZE=4
echo Using GPUs: ${CUDA_VISIBLE_DEVICES}

if [[ $1 == "0" ]]; then
    echo "TRAINING THE BASELINE"
    python models/train_rels.py -m sgcls -model motifnet -nl_obj 0 -nl_edge 0 -b 6 \
    -clip 5 -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/vgdet/vg-faster-rcnn.tar -save_dir checkpoints/baseline2 \
    -nepoch 50 -use_bias
elif [[ $1 == "1" ]]; then
    echo "TRAINING MESSAGE PASSING"
    python models/train_rels.py -m sgcls -model stanford -b 6 -p 100 -lr 1e-3 -ngpu 1 -clip 5 \
    -ckpt checkpoints/vgdet/vg-24.tar -save_dir checkpoints/stanford2
elif [[ $1 == "2" ]]; then
    echo "TRAINING MOTIFNET"
    python models/train_rels.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 2 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 \
        -save_dir checkpoints/motifnet2 -nepoch 50 -use_bias -ckpt checkpoints/vgdet/vg-faster-rcnn.tar
elif [[ $1 == "3" ]]; then
    echo "TRAINING FCKNET v1"
    python models/train_rels.py -m sgcls -model fcknet_v1 -b 8 -p 100 -lr 1e-3 -ngpu ${NUM_GPU} -clip 5 \
    -ckpt checkpoints/fcknet-sgcls/vgrel-4.tar \
    -save_dir checkpoints/fcknet-sgcls
elif [[ $1 == "4" ]]; then
    echo "TRAINING FCKNET V2"
    python models/train_rels.py -m sgcls -model fcknet_v2 -b ${BATCH_SIZE} -p 100 -lr 1e-3 -ngpu ${NUM_GPU} -clip 5 \
    -use_tf -nwork 2 \
    -ckpt checkpoints/fcknet_v2-sgcls/vgrel-8.tar \
    -save_dir checkpoints/fcknet_v2-sgcls
elif [[ $1 == "5" ]]; then
    echo "TRAINING FCKNET V3"
    python models/train_rels.py -m sgcls -model fcknet_v3 \
    -b ${BATCH_SIZE} -p 100 -lr 1e-3 -ngpu ${NUM_GPU} -clip 5 \
    -use_tf -nwork 4 \
    -ckpt checkpoints/vgdet/vg-faster-rcnn.tar \
    -save_dir checkpoints/fcknet_v3-sgcls
else
    usage
fi

#-ckpt checkpoints/motifnet2/vgrel-9.tar \
#-ckpt checkpoints/fcknet-sgcls-wo-batchnorm/vgrel-6.tar \
#y-ckpt checkpoints/vgdet/vg-faster-rcnn.tar \
#-ckpt checkpoints/fcknet_v2-sgcls/vgrel-0.tar \
#-ckpt checkpoints/fcknet_v2-sgcls/vgrel-8.tar

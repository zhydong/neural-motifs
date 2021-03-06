#!/usr/bin/env bash

# This is a script that will evaluate all models for SGCLS
CUDA_VISIBLE_DEVICES=8
echo Using GPUs: ${CUDA_VISIBLE_DEVICES}

if [[ $1 == "0" ]]; then
    echo "EVALING THE BASELINE"
    python models/eval_rels.py -m sgcls -model motifnet -nl_obj 0 -nl_edge 0 -b 6 \
    -clip 5 -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/baseline-sgcls/vgrel-11.tar \
    -nepoch 50 -use_bias -test -cache baseline_sgcls
    python models/eval_rels.py -m predcls -model motifnet -nl_obj 0 -nl_edge 0 -b 6 \
    -clip 5 -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/baseline-sgcls/vgrel-11.tar \
    -nepoch 50 -use_bias -test -cache baseline_predcls
elif [[ $1 == "1" ]]; then
    echo "EVALING MESSAGE PASSING"
    python models/eval_rels.py -m sgcls -model stanford -b 6 -p 100 -lr 1e-3 -ngpu 1 -clip 5 \
    -ckpt checkpoints/stanford-sgcls/vgrel-11.tar -test -cache stanford_sgcls
    python models/eval_rels.py -m predcls -model stanford -b 6 -p 100 -lr 1e-3 -ngpu 1 -clip 5 \
    -ckpt checkpoints/stanford-sgcls/vgrel-11.tar -test -cache stanford_predcls
elif [[ $1 == "2" ]]; then
    echo "EVALING MOTIFNET"
    python models/eval_rels.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/motifnet2/vgrel-5.tar -nepoch 50 -use_bias -cache motifnet_sgcls \
        -multipred
    python models/eval_rels.py -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/motifnet2/vgrel-5.tar -nepoch 50 -use_bias -cache motifnet_predcls
elif [[ $1 == "3" ]]; then
    echo "Evaling FckNet"
    python models/eval_rels.py -m sgcls -model fcknet_v1 -b 6 -ngpu 1 \
        -ckpt checkpoints/fcknet-sgcls/vgrel-4.tar -test -cache fcknet-sgcls
    python models/eval_rels.py -m predcls -model fcknet_v1 -b 6 -ngpu 1 \
        -ckpt checkpoints/fcknet-sgcls/vgrel-4.tar -test -cache fcknet-predcls
elif [[ $1 == "4" ]]; then
    echo "Evaling FckNet v2"
    python models/eval_rels.py -m sgcls -model fcknet_v2 -b 6 -ngpu 1 \
        -ckpt checkpoints/fcknet_v2-sgcls/vgrel-7.tar -test #-cache fcknet-sgcls
    python models/eval_rels.py -m predcls -model fcknet_v2 -b 6 -ngpu 1 \
        -ckpt checkpoints/fcknet_v2-sgcls/vgrel-0.tar -test #-cache fcknet-predcls
elif [[ $1 == "5" ]]; then
    echo "Evaling FckNet v3"
    python models/eval_rels.py -m sgcls -model fcknet_v3 -b 6 -ngpu 1 \
        -ckpt checkpoints/fcknet_v3-sgcls/vgrel-0.tar -test #-cache fcknet-sgcls
    python models/eval_rels.py -m predcls -model fcknet_v3 -b 6 -ngpu 1 \
        -ckpt checkpoints/fcknet_v3-sgcls/vgrel-0.tar -test #-cache fcknet-predcls
fi


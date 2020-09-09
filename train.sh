#!/usr/bin/env bash
A="$1"
B="$2"
echo "The dataset is $B and we will start $A ing"
if [ $A == 'train' ] && [ $B == 'regdb' ]
then
    python main.py --dataset_path '/home/chris/research_work/Datasets/RegDB' --dataset 'regdb'  --trial 1 --mode train
elif [ $1 == 'train' ] && [ $2 == 'sysu' ]
then
    python main.py --dataset_path '/home/chris/research_work/Datasets/SYSU-MM01'  --mode train
elif [ $1 == 'test' ] && [ $2 == 'regdb' ]
then
    python main.py --dataset_path '/home/chris/research_work/Datasets/RegDB' --dataset 'regdb'  --trial 1 --mode test --pretrained_model_epoch 74
elif [ $1 == 'test' ] && [ $2 == 'sysu' ]
then
    python main.py --dataset_path '/home/chris/research_work/Datasets/SYSU-MM01' --dataset 'sysu'  --trial 1 --mode test
fi


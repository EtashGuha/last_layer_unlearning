#!/bin/sh

python exps/disagreement.py -c cfgs/waterbirds.yaml --seed 1 --disagreement_set val --disagreement_proportion 0.5
#python groundzero/main.py -c cfgs/celeba.yaml --seed 1 --max_epochs 50 --accumulate_grad_batches 4
#python groundzero/main.py -c cfgs/waterbirds.yaml --seed 1 --max_epochs 100

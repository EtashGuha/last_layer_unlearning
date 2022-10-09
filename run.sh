#!/bin/sh

python groundzero/main.py -c cfgs/waterbirds.yaml --seed 1 --balanced_sampler True --weights lightning_logs/version_405/checkpoints/last.ckpt --resume_training True
#python exps/disagreement.py -c cfgs/waterbirds.yaml --seed 1 --disagreement_set val --disagreement_proportion 0.5
#python groundzero/main.py -c cfgs/celeba.yaml --seed 1 --max_epochs 50 --accumulate_grad_batches 4
#python groundzero/main.py -c cfgs/waterbirds.yaml --seed 1 --max_epochs 100

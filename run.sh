#!/bin/sh

python exps/disagreement.py -c cfgs/waterbirds.yaml --seed 42 --max_epochs 5 --disagreement_set val --disagreement_proportion 0.5

# Ground Zero
## Quick and extendable experimentation with vision models for classification

### Setup
```
conda update -n base -c defaults conda
conda create -n groundzero python==3.10
conda activate groundzero
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
python -m pip install -e .
```

### Workflow
Base code goes in `groundzero`. Experiment code goes in `exps`. Config files go in `cfgs`. When submitting a new experiment through Condor, rewrite the line in `run.sh` to match the experiment you want to run.

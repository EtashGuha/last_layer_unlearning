# Ground Zero: Quick and extendable experimentation

### Setup
```
conda update -n base -c defaults conda
conda create -n testbed python==3.10
conda activate testbed
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
python -m pip install -e .
condor_submit run.cfg
```

### Workflow
Base code goes in `groundzero`. Experiment code goes in `exps`. Config files go in `cfgs`. When submitting a new experiment through Condor, rewrite the line in `run.sh` to match the experiment you want to run.

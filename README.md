# Ground Zero
## Quick and extendable experimentation with classification models

### Setup
```
conda update -n base -c defaults conda
conda create -n groundzero python==3.10
conda activate groundzero
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
python -m pip install -e .
```

### Workflow
Base code goes in `groundzero`. Experiment code goes in `exps`. Config files go in `cfgs`.

### TODOs
Make lazy loading option where self.data is strings (like in FMOW).

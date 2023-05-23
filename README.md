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
1. Make lazy loading option where self.data is strings (like in FMOW).
2. Upgrade to PyTorch 2.0 and Lightning 2.0
3. Possible W&B integration

## FAQs and gotchas
1. The variable `args` is used for the dictionary of model parameters, but this unfortunately collides with `*args`, the typical Python variable for an unspecified number of positional arguments. Therefore, `*xargs` is used for positional arguments instead.
2. While PyTorch Lightning handles setting many random seeds, one should still use default_rng(seed=args.seed) or Generator().manual_seed(self.seed), especially in DataModules. This is especially important when splitting the dataset so that the random split remains constant even when running multiple training loops.

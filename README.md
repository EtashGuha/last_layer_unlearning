# Ground Zero
## Quick and extendable experimentation with classification models

### Installation
```
conda update -n base -c defaults conda
conda create -n groundzero python==3.10
conda activate groundzero
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
python -m pip install -e .
```

### Instructions
The `groundzero` folder contains the base code and is written in PyTorch Lightning 1.7. Three important files are `groundzero/main.py`, which runs experiments, `groundzero/datamodules/datamodule.py`, which includes data processing and loading, and `groundzero/models/model.py`, which includes model training and inference. These files typically should not need to be modified for experimentation, unless a new basic functionality is being added.

The `cfgs` folder contains configuration files in the `yaml` language which specify training and model parameters. In addition to the options in `groundzero/args.py`, all PyTorch Lightning 1.7 Trainer variables are valid configuration parameters.

The `exps` folder contains experiment code and is where most new code should go. Each experiment in `exps` should have its own `main` method, and the standard workflow is to subclass Models or DataModules as required. For example, `exps/adversarial.py` subclasses `CNN` and `ResNet`, then overrides their `step` function to implement adversarial training.

To run an experiment, pass the main file (either `groundzero/main` or some file in `exps`) with the configuration specified with `-c`. For example,

`python groundzero/main.py -c cfgs/mnist.yaml`

To change parameters, one can either write a new configuration file, or pass variables on the command line:

`python groundzero/main.py -c cfgs/mnist.yaml --lr 0.5`

Downloaded data should go in the `data` folder and outputs (e.g., plots) should go in the `out` folder. The model outputs will be saved to `lightning_logs`. I like to redirect my experiment output from `stdout` to a file in the `logs` folder, but this isn't strictly necessary.

### TODOs
1. Make lazy loading option where self.data is a list of strings.
2. Possible W&B integration.
3. Fix MNIST to subclass from Dataset.
4. Change worst_class_acc to show all class accuracies and make it an option.

## FAQs and gotchas
1. The variable `args` is used for the configuration dictionary, but this unfortunately collides with `*args`, the typical Python variable for an unspecified number of positional arguments. Therefore, `*xargs` is used for positional arguments instead.
2. While PyTorch Lightning handles setting many random seeds, one should still use `np.random.default_rng(seed=args.seed)` or `Generator().manual_seed(self.seed)`, especially in DataModules. This is especially important when splitting the dataset so that the random split remains constant even when running multiple training loops.
3. All new datasets should inherit from `groundzero.datamodules.dataset.Dataset`, and all new models should inherit from `groundzero.models.model.Model`. This may require writing a new class for the dataset/model, even if you are importing it from somewhere else (see `groundzero/datamodules/mnist.py` for an example).
4. This codebase is named after the Ground Zero Performance Cafe at USC, where I did my undergrad. It was demolished to make room for new residence halls, but the memory of their milkshakes will live on forever.

"""Import functions for groundzero."""

# Imports groundzero packages.
import groundzero
from groundzero.datamodules import *
from groundzero.models import *


def valid_model_and_datamodule_names():
    """Returns valid input names for models and datamodules."""

    model_names = [n for n in groundzero.models.__all__ if n != "model"]
    datamodule_names = [n for n in groundzero.datamodules.__all__ if n not in ("dataset", "datamodule")]

    return model_names, datamodule_names

def valid_models_and_datamodules():
    """Returns {name: class} dict for valid models and datamodules."""

    model_names, datamodule_names = valid_model_and_datamodule_names()

    models = [groundzero.models.__dict__[name].__dict__ for name in model_names]
    models = [dict((k.lower(), v) for k, v in d.items()) for d in models]
    models = {name: models[j][name.replace("_", "")] for j, name in enumerate(model_names)} 

    datamodules = [groundzero.datamodules.__dict__[name].__dict__ for name in datamodule_names]
    datamodules = [dict((k.lower(), v) for k, v in d.items()) for d in datamodules]
    datamodules = {name: datamodules[j][name.replace("_", "")] for j, name in enumerate(datamodule_names)} 
    return models, datamodules


# The Ocean Cleanup on Azure ML

This repo contains components for The Ocean Cleanup to work with Azure ML.
There is a package called `toc_azurewrapper` that provides some helpful
functions, meaning the amount of direct interaction with AzureML is decreased
and consistency is increased.

Furthermore, there are skeleton files for getting started with both model
training and model deployment, as well as example implementations of these and
notebooks outlining how to perform these actions.

## Repository Structure

There are a few seperate directories:

- `AzureML_API_examples/`:  These are some basic scripts showing basic
                            operations on AzureML. These are superseded by /
                            implemented in the operations in the
                            `toc_azurewrapper` package, and are left here
                            merely as reference.
- `ModelDeployment/`:       Examples, skeleton files and notebooks for
                            deploying an existing model for production. See the
                            included README in that directory for more
                            information.
- `ModelTraining/`:         Examples, skeleton files and notebooks for training
                            a model. See the included README in that directory
                            for more information.
- `toc_azurewrapper/`:      Package providing often-used Azure ML operations.
                            See below for instructions on how to install.

# Getting started

## Installing AzureML Wrapper

To get started, first we need to install the Azure wrapper. This will also
install the AzureML SDK, if not yet installed.

Navigate to the `toc_azurewrapper` directory, and install the package through
pip:

```
cd toc_azurewrapper
pip install .
```

After this, verify that the package is installed correctly by opening a new
terminal and executing `python -c "import toc_azurewrapper"`. If this gives no
errors, everything is installed correctly.

## Training a model

Training a model is done with the resources found in the `ModelTraining`
directory. For a detailed instruction, see the README there. In short, the
process is as follows. First a wrapper around the model needs to be created. A
start can be found in the `ModelTraining/skeleton_files/` directory as
`train.py`, and example implementations are available under
`ModelTraining/examples/`.

Once this wrapper is available, the model can be trained. For this, a few
things need to be specified:

- an `Environment`, which describes the requirements etc.
- a `compute target`, which is the resource where the training is performed
- an `Experiment`, a logical container for multiple training runs
- the desired `Data Sets` for training and testing
- a `ScriptRunConfig`, the total configuration, together with the code of the
  model

After this, the `ScriptRunConfig` can be submitted to the `Experiment` to start
the training run.

Examples and more detailed instructions about how to do all this can be found
as notebooks in the `ModelTraining/` directory,

## Deploying a model

Deploying a model is done with the resources found in the `ModelDeployment/`
directory. See the README there for detailed information. In short, the
process is as follows. First, the model must have been trained and registered.
Once this is done, we need to create a so-called _scoring script_, `score.py`.
This script needs to implement two functions: `init()` and `run(request)`. The
first function will load or initialize the model, the second will handle a
single request for performing inference on an image. As with the model
training, a skeleton version of this file can be found under the
`ModelDeployment/skeleton_files` directory as `score.py`, and example
implementations are available under `ModelDeployment/examples/`.

Once the scoring script is created, like with the training we need to specify
a few things:

- a `Model` to deploy
- an `Environment`, which describes the requirements etc.
- a `compute target`, which is the resource where the inference is performed
- a `Service`, which is the created deployment from the model and environment,
  deployed to the indicated compute target

Examples and more detailed instructions about how to do all this can be found
as notebooks in the `ModelDeployment/` directory,

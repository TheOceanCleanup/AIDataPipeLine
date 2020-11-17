# Training and registering models with Azure ML

This repo provides the required items to train and register models 
within Azure ML.

The subdirectories contain the following:

- `skeleton_files`: These files are basics for creating a new
                    Run within AzureML. `train.py` takes the
                    provided datasets and parameters, and uses
                    these to orchestrate the training of the model.
                    This code is run within the training
                    environment on AzureML, which is either a
                    locally or within Compute cluster on AzureML,
                    but all organized through AzureML.
- `examples`:       These are implementations of the skeleton files
                    for some example models.

# Getting started

When you have a new model that you want to train using AzureML, you
will first make a copy of the `skeleton_files` folder (or from one
of the examples). There are some things to fill in in `train.py`.
`utils.py` contains functions that can help you with this. The 
following things need to be added/changed in these files.

- What parameters your model requires (and you will provide per
  training run)
- How to format/load the datasets so your model can handle it. Some
  functions to do this are available in `utils.py`, but your case
  might require some special handling.
- How to prepare the model and start the training process
- How to analyze the performance of your trained model
- What metrics to store
- What artifacts of the model to store with the run (in order to
  be able to use them with a later deployment)

Examples of implementations of this can be found in the `examples`
directory.

Once you have this file created, you can use the helper functions
from the `azurewrapper` directory to perform Runs within an
Experiment on AzureML. Examples of this can be found in the 
notebooks in the ModelTraining directory.

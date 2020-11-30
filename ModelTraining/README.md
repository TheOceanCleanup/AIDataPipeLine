# Training and registering models with Azure ML

This repo provides the required items to train and register models within Azure
ML.

The subdirectories contain the following:

- `skeleton_files/`:  These files are basics for creating a new Run within
                      AzureML. `train.py` takes the provided datasets and
                      parameters, and uses these to orchestrate the training of
                      the model. This code is run within the training
                      environment on AzureML, which is either a locally or
                      within Compute cluster on AzureML, but all organized
                      through AzureML.
- `examples/`:        These are implementations of the skeleton files for some
                      example models.

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
from the `toc_azurewrapper` package to perform Runs within an
Experiment on AzureML. Examples of this can be found in the 
notebooks in the ModelTraining directory.

# Dataset expectations

In the development of this tooling we have made some assumptions on the
structure of datasets. These are to make integration with the (results
of) the label API as smooth as possible. If you use a dataset that was
not generated through the label API, your dataset structure may differ,
and some custom interacting with the system may be required.

The following assumptions are made:

### All datasets are a combination of an image set and a label set

We assume that for all datasets consist of two sets: An image set,
registered in Azure ML as File Dataset, and a label image set,
registered in Azure ML as Tabular Dataset.

**Impacts**:

- `toc_wrapper.train.create_args`, `toc_wrapper.train.perform_run`:
The arguments are created based on tuples as `(imageset, labelset)`.
- `skeleton_files.utils`: The tools in utils primarily are used to
parse the datasets into the correct form for your model. These all
assume both image- and labelsets are available, and provided in the
way that `toc_wrapper.train.perform_run` does.

### Label set structure

The labelsets are supposed to be a Tabular dataset. The following 
columns are expected:

- `image_url`: Path to the file, starting from the `imagesets` 
directory in the `toctssstorage` container.
- `label`: A dump of a list of Python objects, each containing the 
following properties:

  - `type`: Name of the detected object
  - `bottomX`: Start of x range of bounding box
  - `topX`: End of x range of bounding box
  - `bottomY`: Start of y range of bounding box
  - `topY`: End of x range of bounding box

  The label API produces these objects through a CSV dump of a
DataFrame. They are read in through `ast.literal_eval`.
- `label_confidence`: The confidence of each label. List of floating 
point values, passed in the same order as the objects in the `label` 
column.

**Impacts**:

- `skeleton_files.utils`

### Image set structure

The imagesets are expected to be a File Dataset. The structure of these
is somewhat out of our control. In practice, we have found that a set
created through `Dataset.File.from_files()`, as we do in the label API,
will be mounted for training in an additional subdirectory indicating
the Datastore the files are a part of. This is configured for `utils`
through the environment variable `DATASTORE_NAME`. The functions in
`utils` will therefore insert these, making the file path for each
image (and therefore the relation to the labels in the labelset) as
follows:

`<mount location>/{DATASTORE_NAME}/{label['image_url']}`

The mount location is injected by Azure into the arguments of the train
script, and will be something like `/tmp/<some random folder name>`.

**NOTE**: when creating a dataset through ML studio, we have found that
the Datastore is not part of the filepath. Therefore, in this case, 
the code needs to be modified to load images from a path as follows:

`<mount location>/{label['image_url']}`

**Impacts**:

- `skeleton_files.utils`
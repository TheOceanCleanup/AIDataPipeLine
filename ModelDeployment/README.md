# Deploying Models to Azure ML

This repo provides the required items to deploys models to Azure ML.

The subdirectories contain the following:

- `skeleton_files/`:  These files are basics for creating a deployment within
                      AzureML. `score.py` implements two functions: `init()`
                      and `run(request)`. The first function will load or
                      initialize the model, the second will handle a single
                      request for performing inference on an image.
- `examples/`:        These are implementations of the skeleton files for some
                      example models.

# Getting started

## Creating a compute target

Models can be deployed to three different systems:

- locally: By not providing a compute target, the endpoint will be created
  locally. This is great for debugging and testing. Once the endpoint is
  running, changes made to the scoring script can be applied by running
  `service.reload()`, allowing for faster iterations. When using the
  `toc_azurewrapper.deploy.deploy()`, a local deployment can be done by either
  ommitting the `target` parameter or setting it to `local`.
- on ACI: By providing `target="aci"`, the model can be deployed to an Azure
  Container Image. This is useful for deploying the model on Azure, when there
  is no Kubernetes cluster available (yet).
- on AKS: By providing `target="aks"` and
  `compute_target_name="<cluster name>"`, where `cluster name` is the name of
  the cluster in Azure ML (see below), the model is deployed to a Kubernetes
  cluster. In this case, GPUs may be used, and scaling options are available.
  This is the method that should be used for production deployments.

### Creating an inference cluster

In order to perform inference on AKS, a cluster first needs to be available
within Azure ML. This can be done best from the Azure ML Studio. You can either
create a new cluster, or attach an existing one to the Azure ML workspace.

To do so, go to `Compute` in the `Manage` section on the left bar in Azure ML
studio. Then, select the tab `Inference Clusters`. Here, click `New`. If you
want to attach an existing cluster, select _Use Existing_ and select the
cluster from the list. Then click _Create_, specify a name by which to find and
use the cluster in Azure ML, and finish the operation.

If you want to create a new cluster, select _Create new_ instead. Then, choose
the desired region (TOC uses `North Europe`). Subsequently, choose the desired
VM size. Note a few things here:
- Not all VM types are available in North Europe, some trial and error may be
  involved.
- Note the number of cores available on the VM. If you wish to create the
  cluster as `production` cluster, at least a total of 12 cores will be
  required. You can choose how many of these VM's there are in your cluster in
  the next step. See the bottom of this readme for the differences between a
  `dev-test` cluster and a `production` cluster.

Now click _Next_. Now specify a name by which to find and use the cluster in
Azure ML. Then, select whether this cluster is a `dev-test` cluster, or a
`production` one. Nex, choose the number of nodes in the cluster. Optionally,
specify the virtual network or enable SSL.

Now click _Create_, and the cluster will be provisioned.

## Creating the scoring script.

When want to deploy a new model after training and registering it, you
will first make a copy of the `skeleton_files` folder (or from one
of the examples). You will now have to fill in the `init()` and `run(request)`
functions in the `score.py` script.

In the `init()` function, you will be loading the artifacts that were
registered with the model, as well as performing any other initialization
logic. The artifacts are mounted on the filesystem as a folder, pointed to by
the environment variable.

In the `run(request)` function, you receive the raw binary bytes of the image.
You will then implement some way to format this data as required by your model
(some often used examples are given). Then, you can perform any required
preprocessing, then perform the inference using your model, performing any 
postprocessing if required, and finally you will form the result to a JSON
structured output.

The output should be structured like this: a list of itms. Each item represents
one detected object, with the properties `xmin`, `xmax`, `ymin` and `ymax` for
the bounding box, `label` containing the (textual) label that was detected, and
optionally a score or confidence metric.

For example:
```
[
  {
    "xmin": 1205,
    "xmax": 1250,
    "ymin": 675,
    "ymax": 726,
    "score": 0.8589632511138916,
    "label": "plastic"
  },
  {
    "xmin": 1357,
    "xmax": 1394,
    "ymin": 980,
    "ymax": 1016,
    "score": 0.8143782019615173,
    "label": "plastic"
  }
]
```

## Deploying the model

Once you have this file created, you can use the helper functions
from the `toc_azurewrapper` package to perform Runs within an
Experiment on AzureML. Examples of this can be found in the 
notebooks in the ModelTraining directory.

# Azure ML on AKS: dev-test vs production

When deploying a cluster as `production` cluster, there is a minimum
requirement of at least 12 cores in total in the cluster. This has an impact on
the costs of the cluster, especially in the beginning stages where we don't run
many endpoints yet.

A `dev-test` cluster does not have this limitation. Therefore, we will now
discuss the differences between these cluster types, so you can decide whether
it is an option to run a `dev-test` cluster instead.

https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-kubernetes?tabs=python poses the following difference:

> A dev-test cluster is not suitable for production level traffic and may
  increase inference times. Dev/test clusters also do not guarantee fault
  tolerance.

This means that the cluster may not be able to handle as much traffic in
`dev-test` mode, and it does not have the same level of fault tolerance.
However, if some increased queues or some manual reapplying of messages from
the dead-letter queue is acceptable, then this is not a disqualifying option.

https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-azure-kubernetes-service?tabs=python poses the following
difference:

> When using a cluster configured as dev-test, the self-scaler is disabled.

This means that autoscaling will not work. If it is okay to manually set the
number of replicas for a deployment (through the `num_replicas` setting of
`AksWebservice.deploy_configuration()`, used in `toc_azurewrapper/deploy.py`),
this would also be okay.

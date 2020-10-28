from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset
from azureml.data import FileDataset, TabularDataset


def create_experiment(workspace, experiment_name):
    return Experiment(workspace=workspace, name=experiment_name)


def _create_args(datasets={}, parameters={}):
    args = []

    for k, v in datasets.items():
        args.append(f"--{k}")
        if type(v) == FileDataset:
            v = v.as_named_input(k).as_mount()
        else:
            v = v.as_named_input(k)
        args.append(v)

    for k,v in parameters.items():
        args.append(f"--{k}")
        args.append(v)

    return args


def perform_run(experiment, script, source_directory, environment=None,
        compute_target=None, datasets={}, parameters={}):

    if environment is None:
        environment = Environment("user-managed-env")
        environment.python.user_managed_dependencies = True

    args = _create_args(datasets, parameters)

    # No compute target is provided, hence the Run is performed locally
    src = ScriptRunConfig(
        source_directory=source_directory,
        script=script,
        arguments=args,
        environment=environment
    )

    run = experiment.submit(config=src)
    return run

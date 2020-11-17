from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset
from azureml.data import FileDataset, TabularDataset


def create_experiment(workspace, experiment_name):
    return Experiment(workspace=workspace, name=experiment_name)


def _create_args(trainsets=[], testsets=[], parameters={}):
    args = []

    args.append("--train_sets")
    for i, (labels, images) in enumerate(trainsets):
        lab = labels.as_named_input(f'train_labels_{str(i)}')
        img = images.as_named_input(f'train_images_{str(i)}').as_mount()
        args.append(lab)
        args.append(img)

    args.append(f"--test_sets")
    for i, (labels, images) in enumerate(testsets):
        lab = labels.as_named_input(f'test_labels_{str(i)}')
        img = images.as_named_input(f'test_images_{str(i)}').as_mount()
        args.append(lab)
        args.append(img)

    for k,v in parameters.items():
        args.append(f"--{k}")
        args.append(v)

    return args


def perform_run(experiment, script, source_directory, environment=None,
        compute_target=None, trainsets=[], testsets=[], parameters={},
        distributed_job_config=None):

    if environment is None:
        environment = Environment("user-managed-env")
        environment.python.user_managed_dependencies = True

    args = _create_args(trainsets, testsets, parameters)

    # No compute target is provided, hence the Run is performed locally
    src = ScriptRunConfig(
        source_directory=source_directory,
        compute_target=compute_target,
        script=script,
        arguments=args,
        environment=environment,
        distributed_job_config=distributed_job_config
    )

    run = experiment.submit(config=src)
    return run

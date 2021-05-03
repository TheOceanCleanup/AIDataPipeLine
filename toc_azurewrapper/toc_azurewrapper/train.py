# AIDAtaPipeLine - A series of examples and utilities for Azure Machine Learning Services
# Copyright (C) 2020-2021 The Ocean Cleanup™
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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

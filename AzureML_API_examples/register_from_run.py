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

# Register a model from a successful run in a given experiment
from azureml.core.model import Model
from azureml.core import Experiment
from azureml.core import Run
from workspace import get_workspace

workspace = get_workspace()

# Select the experiment
experiment = Experiment(workspace=workspace, name='test_experiment_1')

# Get all the runs for the experiment. Returns a generator, which yields the
# runs in reverse chronological order - ie the latest run first. Here, we
# simply select the latest run
runs = experiment.get_runs()
run = next(runs)

# Register the model from the run with a name, some tags and some properties
run.register_model(
    model_name='model_from_test_experiment_1',
    tags={'tag1': 'v1'},
    properties={'property1': 'p1'},
    model_path='outputs/churn-model-2.pkl')
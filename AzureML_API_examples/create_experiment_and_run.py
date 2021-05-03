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

# Create a new experiment, environment (uncontrolled in this case) and use these
# for a training run, using the 'images' dataset. It trains the model through
# the training script train/train.py.
#
# Note that the model that is being trained in the example is very basic and
# does not actually use the provided --regularization parameter, nor does it
# provide an actual performance result. These are merely hardcoded to show the
# effect in AzureML.
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset
from workspace import get_workspace

workspace = get_workspace()


experiment_name = 'test_experiment_1'
experiment = Experiment(workspace=ws, name=experiment_name)


myenv = Environment("user-managed-env")
myenv.python.user_managed_dependencies = True

dataset = Dataset.get_by_name(ws, name='images')

args = ['--data-folder', dataset.as_mount(), '--regularization', 0.07]

# No compute target is provided, hence the Run is performed locally
src = ScriptRunConfig(source_directory='model_train',
                      script='train.py',
                      arguments=args,
                      environment=myenv)

run = experiment.submit(config=src)
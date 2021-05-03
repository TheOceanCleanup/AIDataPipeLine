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

# Deploy an existing model from the model repository, in this case to a local
# service (ie a local docker container)
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig
from azureml.core import Model
from azureml.core.webservice import LocalWebservice
from workspace import get_workspace

workspace = get_workspace()

# Create an environment based on the AzureML-Minimal environment
env = Environment.get(workspace=workspace, name="AzureML-Minimal").clone("TestEnv")

# Add packages manually. This can also be done automatically from either a
# conda dependencies export or a pip requirements file
for pip_package in ["scikit-learn"]:
    env.python.conda_dependencies.add_pip_package(pip_package)

# Create the inverence config from the deploy/ folder
inference_config = InferenceConfig(
    entry_script='deploy/main.py',
    environment=env)

# Use the following model from the repository
model = Model(workspace, 'model_from_test_experiment_1')

service = Model.deploy(
    workspace = workspace,
    name = "my-web-service",
    models = [model],
    inference_config = inference_config,
    deployment_config = LocalWebservice.deploy_configuration(port=8890))

# API is available here
print(service.scoring_uri)
print(service.swagger_uri)

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

from azureml.core import Model
from azureml.core.model import InferenceConfig
from azureml.core.webservice import LocalWebservice, Webservice, \
    AciWebservice, AksWebservice
from azureml.exceptions import WebserviceException


def deploy(workspace, name, model, script, source_directory, environment=None,
        target='local', cpu_cores=1, memory_gb=1, compute_target_name=None):
    inference_config = InferenceConfig(
        entry_script=script,
        source_directory=source_directory,
        environment=environment
    )

    if target == 'local':
        deployment_config = LocalWebservice.deploy_configuration(port=8890)
    elif target == 'aci':
        deployment_config = AciWebservice.deploy_configuration(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb
        )
    elif target == 'aks':
        if compute_target_name is None:
            print("compute_target_name required when target='aks'")
            return None
        deployment_config = AksWebservice.deploy_configuration(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            compute_target_name=compute_target_name,
            auth_enabled=False
        )

    try:
        service = Webservice(workspace, name)
    except WebserviceException:
        service = None

    if service is None:
        service = Model.deploy(
            workspace,
            name,
            [model],
            inference_config,
            deployment_config
        )
    else:
        print(
            "Existing service with that name found, updating InferenceConfig\n"
            "If you meant to redeploy or change the deployment option, first "
            "delete the existing service."
        )
        service.update(
            models=[model],
            inference_config=inference_config)
    return service

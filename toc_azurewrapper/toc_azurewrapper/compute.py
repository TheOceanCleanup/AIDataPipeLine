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

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException


def get_compute(workspace, cluster_name, vm_size='STANDARD_NC6', max_nodes=4):
    """
    Get or create a compute cluster. If a cluster with the provided name
    already exists in this workspace, return it. Otherwise, create a new one.

    :param workspace:       The Azure ML workspace to use.
    :param cluster_name:    Name of the cluster to find or create.
    :param vm_size:         Type/size of VM to create on AzureML, if no cluster
                            was found.
    :param max_nodes:       Max number of nodes to give to this cluster.
    :returns:               A ComputeTarget object.
    """
    try:
        compute_target = ComputeTarget(workspace=workspace, name=cluster_name)
        print('Found existing compute target')
    except ComputeTargetException:
        print('Creating a new compute target...')
        compute_config = AmlCompute.provisioning_configuration(
            vm_size=vm_size,
            max_nodes=max_nodes
        )

        compute_target = ComputeTarget.create(
            workspace, cluster_name, compute_config)

        compute_target.wait_for_completion(
            show_output=True, min_node_count=None, timeout_in_minutes=20)

    return compute_target

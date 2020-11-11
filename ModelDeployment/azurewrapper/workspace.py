from azureml.core import Workspace


def get_workspace(subscription_id, resource_group, workspace_name):
    return Workspace(subscription_id, resource_group, workspace_name)
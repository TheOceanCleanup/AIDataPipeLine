from azureml.core import Workspace

# TODO: Fill this in
subscription_id = ''
resource_group = ''
workspace_name = ''

def get_workspace():
    return Workspace(subscription_id, resource_group, workspace_name)
from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication


def get_workspace(subscription_id, resource_group, workspace_name, tenant_id=None):
    if tenant_id is None:
        return Workspace(subscription_id, resource_group, workspace_name)
    else:
        interactive_auth = InteractiveLoginAuthentication(
            tenant_id=tenant_id)
        return Workspace(subscription_id, resource_group, workspace_name,
                         auth=interactive_auth)
# Example how to get a Datastore object from the workspace
from azureml.core import Datastore
from workspace import get_workspace

workspace = get_workspace()

datastore = Datastore.get(workspace, 'new_images_1')


print(datastore.get_path())
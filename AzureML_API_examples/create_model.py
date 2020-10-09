# Register a model directly, not from an existing Run
from azureml.core.model import Model
from workspace import get_workspace

workspace = get_workspace()

model = Model.register(
    workspace=workspace,
    model_path="churn-model.pkl",
    model_name="churn-model-test",
    tags={'tag1': 'v1'},
    properties={'property1': 'p1'}
)
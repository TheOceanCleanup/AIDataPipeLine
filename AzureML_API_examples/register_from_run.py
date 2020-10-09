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
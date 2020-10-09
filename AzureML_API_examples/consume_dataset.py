# Load a (tabular) dataset into pandas
from azureml.core import Dataset
from workspace import get_workspace
import azureml.contrib.dataset

workspace = get_workspace()

dataset = Dataset.get_by_name(workspace, name='test_labeling_20201006_130456')
df = dataset.to_pandas_dataframe()
print(df)
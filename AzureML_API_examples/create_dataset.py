# Create a dataset with images from two separate datastores
# (these have to exist already)
from azureml.core import Dataset, Datastore
from workspace import get_workspace

workspace = get_workspace()


datastore1 = Datastore.get(workspace, "new_images_1")
datastore2 = Datastore.get(workspace, "images")

# create set with specific files
ds = Dataset.File.from_files(
    path=[
        (datastore1, '/subdir1/overflow.jpg'),
        (datastore1, '/subdir2/meltdown.png'),
        (datastore2, '/images/skull.png'),
        (datastore1, '/images/0.png'),
        (datastore1, '/images/skull.png'),
    ]
)

ds.register(
    workspace=workspace,
    name='manual_ds2',
    description='Some manually created DS',
    create_new_version=True
)
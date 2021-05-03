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
# Example how to get a download URL with a SAS token for an image found in a
# dataset
from azureml.core import Datastore
from azure.storage.blob import BlockBlobService, BlobPermissions
from datetime import datetime, timedelta
from workspace import get_workspace

workspace = get_workspace()

# Path as stored in the dataset (for example through
# dataset.to_path()[<n-th image in the set])
path = '/new_images_1/subdir1/overflow.jpg'
path = path.lstrip('/')

# First part of the path indicates the datastore, the rest the filepath
store_name = path.split('/')[0]
filepath = '/'.join(path.split('/')[1:])

# Get the associated data store
datastore = Datastore.get(workspace, store_name)

# Manual building of URL
url = f'https://{datastore.account_name}.blob.{datastore.endpoint}/{datastore.container_name}/{filepath}'
print(url)

# Automated building of the URL, combined with generatating an SAS token

# TODO insert your account key for the storage account here
block_blob_service = BlockBlobService(
    account_name=datastore.account_name,
    account_key=''
)
token = block_blob_service.generate_blob_shared_access_signature(
    datastore.container_name,
    filepath,
    permission=BlobPermissions.READ,
    expiry= datetime.utcnow() + timedelta(minutes=1)
)

# Use token to generate URL
new_url = block_blob_service.make_blob_url(
    datastore.container_name,
    filepath,
    protocol="https",
    sas_token=token)

print(new_url)
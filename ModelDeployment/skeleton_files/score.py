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

from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse
import os
# TODO: import model/required packages


model = None


def init():
    global model
    # TODO initialize model. Model artifacts can be found under the path
    # indicated by the environment variable AZUREML_MODEL_DIR, use
    # `os.getenv('AZUREML_MODEL_DIR')`
    pass

@rawhttp
def run(request):
    """
    Perform inference on a single image, using the model.

    :param data:    Binary representation of the image
    :returns:       AMLResponse
    """
    # data is a binary representation of an image.
    #
    # TODO: If necessary, manipulate data into the correct form. Examples are:
    # - Open file and write binary data to file:
    #   ```
    #   Path('tmp').mkdir(parents=True, exist_ok=True)
    #   fname = 'tmp/' + \
    #       ''.join(random.choice(string.ascii_lowercase) for i in range(32))
    #   with open(fname, 'wb') as f:
    #       f.write(data)
    #   ```
    # - Create a OpenCV image in memory:
    #   ```
    #   import numpy as np
    #   import cv2
    #
    #   nparr = np.frombuffer(data, np.uint8)
    #   original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #   ```
    # - Create a PIL image in memory:
    #   ```
    #   from PIL import Image
    #   import io
    #
    #   image = Image.open(io.BytesIO(data))
    if request.method == 'POST':
        data = request.get_data(False)

        # TODO: Optionally, do something with the data here to create the
        # correct form of image

        # TODO implement actual model prediction
        result = model.predict(data)
        return AMLResponse(result, 200)
    else:
        return AMLResponse("Method not allowed", 405)

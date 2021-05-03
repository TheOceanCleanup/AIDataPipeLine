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
from model import Model
from pathlib import Path
import os
import random
import string


model = None


def init():
    global model
    # Initialize model
    model = Model.load(os.path.join(
        os.getenv('AZUREML_MODEL_DIR'),
        'model.pkl'
    ))


@rawhttp
def run(request):
    """
    Perform inference on a single image, using the model.

    :param data:    Binary representation of the image
    :returns:       AMLResponse
    """

    if request.method == 'POST':
        data = request.get_data(False)

        # Open file and write binary data to file:
        Path('tmp').mkdir(parents=True, exist_ok=True)
        fname = 'tmp/' + \
            ''.join(random.choice(string.ascii_lowercase) for i in range(32))

        with open(fname, 'wb') as f:
            f.write(data)

        result = model.predict(fname)

        os.remove(fname)

        return AMLResponse(result, 200)
    else:
        return AMLResponse("Method not allowed", 405)

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

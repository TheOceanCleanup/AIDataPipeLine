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

import json
import os
import joblib


def init():
    global model
    # The model is loaded in by Azure ML, the directory is stored as env var.
    # Currently unclear if the name of the model file is automatically provided
    # somehow too. That would be a nice addition.
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'churn-model-2.pkl')
    model = joblib.load(model_path)


def run(raw_data):
    try:
        # Our model requires a specific numpy input type, hence the [[ ]]
        # around the provided input. In real scenarios, you'd probably want to
        # do some parameter validation before passing it on.
        result = model.predict([[json.loads(raw_data)]])

        # You can return any data type, as long as it is JSON serializable. In
        # a real scenario you'd probably create a JSON object with a few
        # properties
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
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

# This is a skeleton file that you can use to implement a training script. This
# includes the functionality required to train a model and register the
# run in Azure ML.
#
# To use this, look for the parts marked with TODO - these need to be adjusted
# or filled in.
# A logging object is available - whatever is logged to this will be registered
# in Azure ML as logs of the model training. Don't confuse this with run.log(),
# that is used to register statistics about the model, both parameters and
# performance metrics.
from azureml.core import Run
from utils import load_args, load_datasets_for_yolo_v5
import subprocess
import logging
import shutil
import os

logger = logging.getLogger('model')
fh = logging.FileHandler('logs/model.log')
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
fh.setLevel('DEBUG')
logger.addHandler(fh)
logger.setLevel('DEBUG')


if __name__ == "__main__":
    # Parse arguments. The arguments contains the (path to the) data and the
    # parameters to the model. Provide the parameters as
    # [<name>, <type>, <default value>]
    # TODO add the required parameters.
    parameters = load_args([
        ['weights', str, '.']
    ])

    dataset_path = load_datasets_for_yolo_v5(
        parameters.train_sets,
        parameters.test_sets
    )

    # Move weights file to correct folder

#     shutil.copy(
#             f"{parameters.weights}",
#             f"yolov5/weights/{parameters.weights}"
#     )

    #### Implement/perform model training ####

    logger.info("Starting training")

    p = subprocess.run(
        [
            "python",
            "train.py",
            "--epochs", "120",
            "--data", '../' + dataset_path,
            "--weights", parameters.weights,
            "--batch-size", "16",
            "--hyp", 'data/hyp.finetune.yaml',
            "--cfg", "models/yolov5l.yaml"
        ],
        cwd="yolov5",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    logger.debug(p.stdout)
    logger.warning(p.stderr)

    logger.info("Finished training")


    #### Determine model performance ####
    logger.debug("Measuring performance")

    #### Register model performance with the run ####

    # Load the run object
    run = Run.get_context()

    # Register the parameters used for training and the performance of the
    # model using run.log()
    logger.debug("Registering performance data")

    res_mapping = {
        'GIoU': 2,
        'Objectness': 3,
        'Classification': 4,
        'Precision': 8,
        'Recall': 9,
        'val GIoU': 12,
        'val Objectness': 13,
        'val Classification': 14,
        'mAP@0.5': 10,
        'mAP@0.5:0.95': 11
    }
    results = {}
    with open('yolov5/runs/exp0/results.txt', 'r') as f:
        for l in f.readlines():
            l = l.rstrip('\n').split()
            for t, index in res_mapping.items():
                if t not in results:
                    results[t] = []

                results[t].append(float(l[res_mapping[t]]))

    for t in res_mapping.keys():
        run.log_list(t, results[t])

    # Write the model file to the outputs/ folder. This is then automatically
    # attached to the Run, and to any models registered from that run
    logger.debug("Writing model data")

    os.makedirs('outputs/weights/', exist_ok=True)
    for f in os.listdir('yolov5/runs/exp0/weights/'):
        shutil.copy(f'yolov5/runs/exp0/weights/{f}', f'outputs/weights/{f}')

    logger.info("Train process finished")

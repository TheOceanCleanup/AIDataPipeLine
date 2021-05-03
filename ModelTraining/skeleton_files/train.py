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
from utils import load_args, find_set, load_set_as_txt
import argparse
import logging

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
        ['regularization_rate', float, 0.01]
    ])

    # TODO: Validate that this version of load_set generates the type of output
    #       that is required for the model.
    train_set, train_labels = load_set_as_txt('train', parameters.train_sets)
    test_set, test_labels = load_set_as_txt('test', parameters.test_sets)

    # TODO: This is an example, adjust as required:
    regularization_rate = parameters.regularization_rate


    #### Implement/perform model training ####

    logger.info("Starting training")

    # TODO Add model here and training here
    model = None

    logger.info("Finished training")


    #### Determine model performance ####
    logger.debug("Measuring performance")

    # TODO: Determine model performance here
    accuracy = 0.97


    #### Register model performance with the run ####

    # Load the run object
    run = Run.get_context()

    # Register the parameters used for training and the performance of the
    # model using run.log()
    logger.debug("Registering performance data")

    # TODO: These are examples, adjust as required
    run.log('regularization rate', np.float(regularization_rate))
    run.log('accuracy', accuracy)

    # Write the model file to the outputs/ folder. This is then automatically
    # attached to the Run, and to any models registered from that run
    logger.debug("Writing model data")

    # TODO: This is an example, adjust as required
    model.save('outputs/model.pkl')

    logger.info("Train process finished")

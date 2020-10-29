from azureml.core import Run, Dataset
from model import Model
import logging
import os
from utils import load_args, find_set, load_set_as_txt

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
    parameters = load_args([
        ['param_a', float, 10.0],
        ['param_b', float, 0.5]
    ])

    param_a = parameters.param_a
    param_b = parameters.param_b

    train_set, train_labels = load_set_as_txt('train', parameters.train_sets)
    test_set, test_labels = load_set_as_txt('test', parameters.test_sets)

    #### Implement/perform model training ####

    logger.info("Starting training")

    # Train the model
    model = Model(param_a, param_b)
    model.train(train_set, train_labels)

    logger.debug(model.labels)

    # logger.info("Finished training")

    #### Determine model performance ####
    logger.debug("Measuring performance")

    # Determine model performance
    correct = 0
    total = 0
    with open(test_set) as f:
        for l in f.readlines():
            logger.debug(l)
            l = l.rstrip('\n')
            filename = l.split(' ')[0]
            labels = l.split(' ')[1:]

            for label in labels:
                if model.predict(filename) == label.split(',')[-1]:
                    correct += 1
                total += 1

    if total == 0:
        accuracy = 0
    else:
        accuracy = correct / total

    logger.debug("Done measuring performance")

    #### Register model performance with the run ####

    # Load the run object
    run = Run.get_context()

    # Register the parameters used for training and the performance of the
    # model using run.log()
    logger.debug("Registering performance data")

    run.log('param a', param_a)
    run.log('param b', param_b)
    run.log('accuracy', accuracy)

    # Write the model file to the outputs/ folder. This is then automatically
    # attached to the Run, and to any models registered from that run
    logger.debug("Writing model data")

    model.save('outputs/model.pkl')

    logger.info("Train process finished")

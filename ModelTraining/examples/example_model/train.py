from azureml.core import Run
from model import Model
import argparse
import logging
import ast
import os

logger = logging.getLogger('model')
fh = logging.FileHandler('logs/model.log')
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
fh.setLevel('DEBUG')
logger.addHandler(fh)
logger.setLevel('DEBUG')


def load_args():
    """
    Load the provided arguments.

    :returns:   ArgumentParser object with parsed arguments.
    """
    # The configuration of the run is being passed on as arguments,
    # lets load those
    parser = argparse.ArgumentParser()

    # Get the folders where the data is mounted
    parser.add_argument(
        '--train_images', type=str, dest='train_images',
        help='Training set images'
    )
    parser.add_argument(
        '--train_labels', type=str, dest='train_labels',
        help='Training set labels'
    )
    parser.add_argument(
        '--test_images', type=str, dest='test_images',
        help='Test set images'
    )
    parser.add_argument(
        '--test_labels', type=str, dest='test_labels',
        help='Test set labels'
    )

    # Get the parameters to the model.
    parser.add_argument(
        '--param_a', type=float, dest='param_a',
        default=0.01, help='param a'
    )
    parser.add_argument(
        '--param_b', type=float, dest='param_b',
        default=0.01, help='param b'
    )

    parser.add_argument(
        '--weights', type=str, dest='weights',
        help='Weights dataset'
    )

    return parser.parse_args()


def labels_to_df(dataset):
    df = dataset.to_pandas_dataframe()
    df['label'] = df['label'].apply(ast.literal_eval)
    return df


if __name__ == "__main__":
    # Parse arguments. The arguments contains the (path to the) data and the
    # parameters to the model
    parameters = load_args()

    param_a = parameters.param_a
    param_b = parameters.param_b

    logger.debug()

    train_images = Run.get_context().input_datasets["train_images"] + '/main_datastore/'
    train_labels = Run.get_context().input_datasets["train_labels"]
    test_images = Run.get_context().input_datasets["test_images"] + '/main_datastore/'
    test_labels = Run.get_context().input_datasets["test_labels"]

    logger.debug(Run.get_context().input_datasets["train_images"])

    #### Implement/perform model training ####

    logger.info("Starting training")

    # Train the model
    model = Model(param_a, param_b)
    model.train(
        weights,
        train_images,
        labels_to_df(train_labels)
    )

    logger.info("Finished training")

    #### Determine model performance ####
    logger.debug("Measuring performance")

    # Determine model performance
    correct = 0
    total = 0
    for index, image_labels in labels_to_df(test_labels).iterrows():
        img = test_images + image_labels['image_url']
        logger.debug(img)

        for l in image_labels['label']:
            if model.predict(img) == l['label']:
                correct += 1
            total += 1

    if total == 0:
        accuracy = 0
    else:
        accuracy = correct / total

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

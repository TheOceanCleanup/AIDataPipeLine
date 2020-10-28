from azureml.core import Run, Dataset
from model2 import Model
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
        '--train_sets', type=str, dest='train_sets', nargs='*',
        help='Training sets. Provide as labels:images,labels:images'
    )
    parser.add_argument(
        '--test_sets', type=str, dest='test_sets', nargs='*',
        help='Test sets. Provide as labels:images,labels:images'
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

    return parser.parse_args()


def labels_to_df(dataset):
    """
    Turn a tabular label dataset into a pandas dataframe. Parse the 'labels'
    column into a Python object.

    :params dataset:    The TabularDataset object to convert.
    :returns:           Pandas dataframe.
    """
    df = dataset.to_pandas_dataframe()
    df['label'] = df['label'].apply(ast.literal_eval)
    return df


def find_set(set_id):
    """
    Find a dataset in the inputs by its ID, required for Tabular Datasets.

    :param set_id:      ID of the dataset to load.
    :returns:           The found dataset.
    """
    return Dataset.get_by_id(Run.get_context().experiment.workspace, set_id)


def load_set(name, sets):
    """
    Load the datasets. Expects a list of (label, image) set combinations.

    This version writes files, as expected by the current model Mats is
    building. In other cases, some other processing may be required.

    TODO: Validate if the output of this function is as required.

    :param name:        Name to give the set output
    :param sets:        A list of (label, image) set combinations
    :returns:           Paths to the generated files.
    """
    labelset = set()
    rows = []
    for label_id, image_folder in zip(sets[0::2], sets[1::2]):
        labels = find_set(label_id)

        for i, l in labels_to_df(labels).iterrows():
            row = [image_folder + '/main_datastore/' + l['image_url']]
            for label_entry in l['label']:
                row.append(','.join([
                    str(label_entry['bottomX']),
                    str(label_entry['bottomY']),
                    str(label_entry['topX']),
                    str(label_entry['topY']),
                    label_entry['label']
                ]))
                labelset.add(label_entry['label'])
            rows.append(' '.join(row) + '\n')

    filepath = f'outputs/{name}.txt'
    with open(filepath, 'w') as f:
        f.writelines(rows)

    labelpath = f'outputs/{name}_labels.txt'
    with open(labelpath, 'w') as f:
        f.writelines([x + '\n' for x in labelset])

    return filepath, labelpath


if __name__ == "__main__":
    # Parse arguments. The arguments contains the (path to the) data and the
    # parameters to the model
    parameters = load_args()

    param_a = parameters.param_a
    param_b = parameters.param_b

    train_set, train_labels = load_set('train', parameters.train_sets)
    test_set, test_labels = load_set('test', parameters.test_sets)

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

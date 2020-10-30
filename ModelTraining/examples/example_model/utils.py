from azureml.core import Run, Dataset
import csv
import argparse
import ast
import os


DATASTORE_NAME = 'main_datastore'


def load_args(parameters):
    """
    Load the provided arguments.

    :param parameters:  List of expected parameters, next to the test- and
                        train sets. Each item in the list is a list like such:
                        [<name>, <type>, <default value>]
    :returns:           ArgumentParser object with parsed arguments.
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
    for p in parameters:
        parser.add_argument(
            f"--{p[0]}",
            type=p[1],
            dest=p[0],
            default=p[2]
        )

    return parser.parse_args()


def labels_to_df(dataset):
    """
    Turn a tabular label dataset into a pandas dataframe. Parse the 'labels'
    column into a Python object.

    :param dataset:     The TabularDataset object to convert.
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


def load_set_as_txt(name, sets):
    """
    Load the datasets. Expects a list of (label, image) set combinations.

    This version writes the output to text files, as expected by the some
    models. In other cases, some other processing may be required. These files
    are written to outputs/, so they are stored with the run and can be
    inspected at a later time.

    :param name:        Name to give the set output
    :param sets:        A list of (label, image) set combinations
    :returns:           Paths to the generated files.
    """
    labelset = set()
    rows = []
    for label_id, image_folder in zip(sets[0::2], sets[1::2]):
        labels = find_set(label_id)

        for i, l in labels_to_df(labels).iterrows():
            row = [f"{image_folder}/{DATASTORE_NAME}/{l['image_url']}"]
            for label_entry in l['label']:
                row.append(','.join([
                    str(label_entry['bottomX']),
                    str(label_entry['topX']),
                    str(label_entry['bottomY']),
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


def load_set_as_csv_pbtxt(name, sets, pbtxt=False):
    """
    Load the datasets. Expects a list of (label, image) set combinations.

    This version writes the output to CSV files, as expected by the some
    models. In other cases, some other processing may be required. These files
    are written to outputs/, so they are stored with the run and can be
    inspected at a later time.

    :param name:        Name to give the set output
    :param sets:        A list of (label, image) set combinations
    :param pbtxt:       Boolean indicating whether to generate pbtxt file with
                        labels
    :returns:           Paths to the generated files.
    """
    labelset = set()
    rows = [['class', 'filename', 'height', 'width', 'xmax', 'xmin', 'ymax', 'ymin']]
    for label_id, image_folder in zip(sets[0::2], sets[1::2]):
        labels = find_set(label_id)

        for i, l in labels_to_df(labels).iterrows():
            for label_entry in l['label']:
                row = [f"{image_folder}/{DATASTORE_NAME}/{l['image_url']}"]
                row = [
                    label_entry['label'],
                    f"{image_folder}/{DATASTORE_NAME}/{l['image_url']}",
                    1080,
                    1920,
                    label_entry['topX'],
                    label_entry['bottomX'],
                    label_entry['topY'],
                    label_entry['bottomY']
                ]
                labelset.add(label_entry['label'])
            rows.append(row)

    filepath = f'outputs/{name}.csv'
    with open(filepath, 'w') as f:
        w = csv.writer(f)

        for row in rows:
            w.writerow(row)

    labelpath = f'outputs/labelmap.pbtxt'
    if pbtxt:
        with open(labelpath, 'w') as f:
            out = ""
            for i, l in enumerate(labelset):
                out += \
                    "item {\n" \
                    f"    id: {str(i + 1)}\n" \
                    f"    name: '{l}'\n" \
                    "}\n"
            f.write(out)

    return filepath, labelpath

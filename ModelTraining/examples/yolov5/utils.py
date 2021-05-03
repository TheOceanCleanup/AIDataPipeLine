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

from azureml.core import Run, Dataset
from PIL import Image
import csv
import argparse
import ast
import os
import shutil


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
    labelset = {}
    rows = []
    for label_id, image_folder in zip(sets[0::2], sets[1::2]):
        labels = find_set(label_id)

        for i, l in labels_to_df(labels).iterrows():
            row = [f"{image_folder}/{DATASTORE_NAME}/{l['image_url']}"]
            for label_entry in l['label']:
                # Create new, numbered, entry of label in labelset
                if label_entry['label'] not in labelset:
                    labelset[label_entry['label']] = len(labelset)

                row.append(','.join([
                    str(label_entry['bottomX']),
                    str(label_entry['bottomY']),
                    str(label_entry['topX']),
                    str(label_entry['topY']),
                    str(labelset[label_entry['label']])
                ]))

            rows.append(' '.join(row) + '\n')

    filepath = f'outputs/{name}.txt'
    with open(filepath, 'w') as f:
        f.writelines(rows)

    labelpath = f'outputs/{name}_labels.txt'
    labelset_sorted = sorted(labelset.items(), key=lambda x: x[1])
    with open(labelpath, 'w') as f:
        f.writelines([str(x[0]) + '\n' for x in labelset_sorted])

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
            fp = f"{image_folder}/{DATASTORE_NAME}/{l['image_url']}"
            im = Image.open(fp)
            width, height = im.size

            for label_entry in l['label']:
                row = [
                    label_entry['label'],
                    fp,
                    height,
                    width,
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


def _calc_bbox_center_fraction(img_width, img_height, topX, bottomX, topY,
        bottomY):
    """
    Calculate yolo v5 representation of Bounding Box, which is different in two
    ways:
    - indicated as (center_x, center_y, width, height)
    - coordinates and sizes are indicated as fraction of image, instead of
      pixels.

    :param img_width:   Width of the image
    :param img_height:  Height of the image
    :param topX:        Right border in x direction, as pixels
    :param bottomX:     Left border in x direction, as pixels
    :param topX:        Bottom border in x direction, as pixels
    :param bottomX:     Top border in x direction, as pixels
    :returns:           List of [
                            X coordinate of center, as fraction,
                            Y coordinate of center, as fraction,
                            width of bounding box, as fraction,
                            height of bounding box, as fraction
                        ]
    """
    box_width = topX - bottomX
    box_height = topY - bottomY
    center_x = bottomX + (box_width / 2)
    center_y = bottomY + (box_height / 2)
    return [
        center_x / img_width,
        center_y / img_height,
        box_width / img_width,
        box_height / img_height
    ]


def _parse_set_yolov5(set_type, sets, labelset):
    """
    Parse a set of sets to move images in the correct folder and create label
    files for yolov5.

    :param set_type:        The type of set (train or test)
    :param sets:            A list of (label, image) set combinations
    :param labelset:        Dict containing the labels and their int-based ID
    :returns:               The adjusted labelset.
    """
    # Build train set
    for label_id, image_folder in zip(sets[0::2], sets[1::2]):
        labels = find_set(label_id)

        for i, l in labels_to_df(labels).iterrows():
            fp = f"{image_folder}/{l['image_url']}"
            im = Image.open(fp)
            width, height = im.size

            # Copy file into correct folder
            img_url = l['image_url'].replace('/', '_')
            target = f'data/{set_type}/images/{img_url}'
            label_target = f'data/{set_type}/labels/' + \
                '.'.join(img_url.split('.')[:-1] + ['txt'])
            os.makedirs(os.path.dirname(target), exist_ok=True)
            os.makedirs(os.path.dirname(label_target), exist_ok=True)
            shutil.copy(fp, target)

            with open(label_target, 'w') as f:
                for label_entry in l['label']:
                    # Create new, numbered, entry of label in labelset
                    if label_entry['label'] not in labelset:
                        labelset[label_entry['label']] = len(labelset)

                    row = [str(labelset[label_entry['label']])] + \
                        [str(x) for x in _calc_bbox_center_fraction(
                            width,
                            height,
                            label_entry['topX'],
                            label_entry['bottomX'],
                            label_entry['topY'],
                            label_entry['bottomY']
                        )]
                    f.write(' '.join(row) + '\n')

    return labelset


def load_datasets_for_yolo_v5(train_sets, test_sets):
    """
    Load the datasets. Expects a list of (label, image) set combinations.

    This version writes the output to CSV files, as expected by the yolo v5
    model. It moves the images all into /train/images and /test/images
    locations, and generates a label .txt file per image in /train/labels and
    /test/labels. It then creates a yaml file pointing to these files.

    :param train_sets:  A list of (label, image) set combinations
    :param test_sets:   A list of (label, image) set combinations
    :returns:           Path to the generated yaml file.
    """
    # Prepare dirs
    os.makedirs('outputs/data/train/images', exist_ok=True)
    os.makedirs('outputs/data/train/labels', exist_ok=True)
    os.makedirs('outputs/data/test/images', exist_ok=True)
    os.makedirs('outputs/data/test/labels', exist_ok=True)
    labelset = {}

    labelset = _parse_set_yolov5("train", train_sets, labelset)
    labelset = _parse_set_yolov5("test", test_sets, labelset)

    labelset_sorted = sorted(labelset.items(), key=lambda x: x[1])

    # Create yaml file
    with open('outputs/data/dataset.yaml', 'w') as f:
        f.write('train: ../data/train/images/\n')
        f.write('val: ../data/test/images/\n')
        f.write('\n')
        f.write(f'nc: {str(len(labelset))}\n')
        f.write('\n')
        f.write(f'names: {[x[0] for x in labelset_sorted]}')

    return 'outputs/data/dataset.yaml'

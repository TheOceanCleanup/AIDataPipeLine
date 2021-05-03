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

import os
import time
from tf2od.model_main_tf2 import tf
from utils import find_set, labels_to_df
import numpy as np
import cv2
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


def detect(sets):

    with open('test_export_images.txt', 'r') as f:
        image_names = [r.strip() for r in f.readlines()]

    category_index = label_map_util.create_category_index_from_labelmap('tf2od/annotations/label_map.pbtxt',
                                                                    use_display_name=True)
    # Load model
    print('Loading model...', end='')
    start_time = time.time()
    detect_fn = tf.saved_model.load('outputs/saved_model/')
    print('Done! Took {} seconds'.format(start_time - time.time()))


    # Detect and save images
    for label_id, image_folder in zip(sets[0::2], sets[1::2]):
        image_list = os.listdir(image_folder)
        df_labels = labels_to_df(find_set(label_id))

        for image_path in image_names:
            if image_path in image_list:
                print('Running inference for {}... '.format(os.path.join(image_folder, image_path)))
                image_np = np.array(Image.open(os.path.join(image_folder, image_path)))
                input_tensor = tf.convert_to_tensor(image_np)
                input_tensor = input_tensor[tf.newaxis, ...]

                detections = detect_fn(input_tensor)

                num_detections = int(detections.pop('num_detections'))
                detections = {key: value[0, :num_detections].numpy()
                               for key, value in detections.items()}
                detections['num_detections'] = num_detections
                detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

                image_np_with_detections = viz_utils.visualize_boxes_and_labels_on_image_array(
                      image_np.copy(),
                      detections['detection_boxes'],
                      detections['detection_classes'],
                      detections['detection_scores'],
                      category_index,
                      use_normalized_coordinates=True,
                      max_boxes_to_draw=None,
                      min_score_thresh=.25,
                      line_thickness=1,
                      agnostic_mode=False,
                      skip_labels=True)

                h, w, _ = image_np.shape
                true_labels = df_labels[df_labels['image_url'] == image_path].iloc[0]['label']
                true_labels = np.array([[tl['bottomY']/h, tl['bottomX']/w, tl['topY']/h, tl['topX']/w] for tl in true_labels])

                image_np_with_detections = viz_utils.visualize_boxes_and_labels_on_image_array(
                      image_np_with_detections.copy(),
                      true_labels,
                      None,
                      None,
                      None,
                      use_normalized_coordinates=True,
                      max_boxes_to_draw=None,
                      line_thickness=1,
                      skip_labels=True,
                      groundtruth_box_visualization_color='Yellow'
                      )

                cv2.imwrite('outputs/detections/' + image_path,
                            cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))

    # To keep track of fps
    n_detect = 0
    t_detect = 0

    # To calculate fps
    print("Calculating fps")
    for label_id, image_folder in zip(sets[0::2], sets[1::2]):
        df_labels = labels_to_df(find_set(label_id))

        for image_path in [img for img in os.listdir(image_folder) if img.endswith('.jpg')]:
            print('Running inference for {}... '.format(os.path.join(image_folder, image_path)))
            image_np = np.array(Image.open(os.path.join(image_folder, image_path)))
            input_tensor = tf.convert_to_tensor(image_np)
            input_tensor = input_tensor[tf.newaxis, ...]

            start_time = time.time()
            detections = detect_fn(input_tensor)
            stop_time = time.time()
            t_detect += stop_time - start_time
            n_detect += 1

    fps = float(n_detect) / t_detect
    print(f"FPS: {fps}")
    return fps

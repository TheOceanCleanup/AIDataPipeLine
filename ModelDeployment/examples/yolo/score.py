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

from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse
from Tensorflow_YOLO.yolov3.utils import Load_Yolo_model_custom, \
    image_preprocess, postprocess_boxes, nms, YOLO_CUSTOM_WEIGHTS
import tensorflow as tf
import cv2
import numpy as np
import os
import json


yolo = None
labels = []
input_size = 416
score_threshold=0.3
iou_threshold=0.45


def init():
    global yolo
    print(os.listdir(os.getenv('AZUREML_MODEL_DIR')))
    yolo = Load_Yolo_model_custom(
        os.path.join(
            os.getenv('AZUREML_MODEL_DIR'),
            'outputs/checkpoint/yolov4_custom_Tiny'
        ),
        os.path.join(
            os.getenv('AZUREML_MODEL_DIR'),
            'outputs/labels.names'
        )
    )
    with open(
            os.path.join(os.getenv('AZUREML_MODEL_DIR'),
                         'outputs/labels.names')) as f:
        for l in f.readlines():
            labels.append(l.rstrip('\n'))


@rawhttp
def run(request):
    """
    Perform inference on a single image, using the model.

    :param data:    Binary representation of the image
    :returns:       AMLResponse
    """

    if request.method == 'POST':
        data = request.get_data(False)
        if len(data) == 0:
            return AMLResponse(
                "No data received - please provide an image as body", 400)

        nparr = np.frombuffer(data, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        pred_bbox = yolo.predict(image_data)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')

        print(bboxes)

        result = []
        for box in bboxes:
            result.append({
                'xmin': int(box[0]),
                'xmax': int(box[2]),
                'ymin': int(box[1]),
                'ymax': int(box[3]),
                'score': float(box[4]),
                'label': labels[int(box[5])]
            })

        return AMLResponse(json.dumps(result), 200)
    else:
        return AMLResponse("Method not allowed", 405)


if __name__ == "__main__":
    init()

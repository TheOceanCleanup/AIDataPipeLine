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
input_size = 416
score_threshold=0.3
iou_threshold=0.45


def init():
    global yolo
    global YOLO_CUSTOM_WEIGHTS
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


@rawhttp
def run(request):
    """
    Perform inference on a single image, using the model.

    :param data:    Binary representation of the image
    :returns:       AMLResponse
    """

    if request.method == 'POST':
        data = request.get_data(False)
        # TODO verify data is provided

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
                'label': int(box[5])
            })

        return AMLResponse(json.dumps(result), 200)
    else:
        return AMLResponse("Method not allowed", 405)


if __name__ == "__main__":
    init()

import sys
import time
import greengrasssdk
import platform
import os
from threading import Timer
import logging
import numpy as np
import tensorflow as tf
from PIL import Image   # pillow
import struct
import glob

logging.basicConfig(format='%(asctime)s|%(name)-8s|%(levelname)s: %(message)s',
                    level=logging.INFO)

# Add system paths to use supporting files needed by tensorflow object detection API
MODEL_BASE = 'build/models/research'
sys.path.append(MODEL_BASE)
sys.path.append(MODEL_BASE + '/object_detection')
sys.path.append(MODEL_BASE + '/slim')


from utils import label_map_util  # from build/models/research/utils


#### Create a client to communicate with AWS console in cloud ###
client = greengrasssdk.client('iot-data')

# Path in the green-grass environment from where we can access models deploed from s3 bucket
PATH_TO_CKPT = '/greengrass-machine-learning/tf/frozen_inference_graph.pb'
# PATH_TO_CKPT = 'models/frozen_inference_graph.pb'   # models included with Lambda function

# label file containing class categories
# PATH_TO_LABELS = 'build/mscoco_label_map.pbtxt'
PATH_TO_LABELS = 'models/mscoco_label_map.pbtxt'  # original from tf github


# Detection threshold
CUT_SCORE = 0.5


class ObjectDetector(object):
    def __init__(self):

        self.detection_graph = self._build_graph()
        self.sess = tf.Session(graph=self.detection_graph)

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=90, use_display_name=True)

        self.category_index = label_map_util.create_category_index(categories)


    def _build_graph(self):

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        return detection_graph

    def detect(self, img_width, img_height, image):
        image_np = np.array(image).reshape(
            (img_height, img_width, 3)).astype(np.uint8)
        image_np_expanded = np.expand_dims(image_np, axis=0)

        graph = self.detection_graph
        image_tensor = graph.get_tensor_by_name('image_tensor:0')
        boxes = graph.get_tensor_by_name('detection_boxes:0')
        scores = graph.get_tensor_by_name('detection_scores:0')
        classes = graph.get_tensor_by_name('detection_classes:0')
        num_detections = graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        boxes, scores, classes, num_detections = map(
            np.squeeze, [boxes, scores, classes, num_detections])

        return boxes, scores, classes.astype(int), num_detections


def detect_objects(img_width, img_height, image):
    print('Got image of size ' + str(img_width) + ' x ' + str(img_height) )
    boxes, scores, classes, num_detections = CLIENT.detect(img_width,
                                                           img_height,
                                                           image)
    res = []
    for i in range(int(num_detections)):
    # for i in range(num_detections):
        score = scores[i]
        if score < CUT_SCORE:
            continue
        cls = classes[i]
        ymin, xmin, ymax, xmax = boxes[i]
        (left, right, top, bottom) = (xmin * img_width, xmax * img_width,
                                      ymin * img_height, ymax * img_height)

        category = CLIENT.category_index[cls]['name']
        #category = cls
        res.append((left, right, top, bottom, cls, category))

    for i in range(len(res)):
        print('   Found ' + str(i) + '.box ' + str(res[i][0]) + ' ' +
              str(res[i][1]) + ' ' + str(res[i][2]) + ' ' + str(res[i][3]) +
              ' of category ' + res[i][5])
        # print(res)

    return res


CLIENT = ObjectDetector()


# When deployed to a Greengrass core, this code will be executed immediately
# as a long-lived lambda function.  The code will enter the infinite while loop
# below.
def greengrass_object_detection_run():

    try:

        for filename in glob.glob('images/*.jpeg'): #assuming jpeg
            # print(filename)
            image=Image.open(filename)
            (im_width, im_height) = image.size
            predictions = detect_objects(im_width, im_height, image)
            # print(str(predictions))
            logging.info("{0}: {1}".format(filename, str(predictions)))

        # publish predictions
        client.publish(topic='hello/world', payload='New Prediction: {}'.format(str(predictions)))
    except Exception as ex:
        e = sys.exc_info()[0]
        print("Exception occured during prediction: %s" % e)
        print("Ex: %s" % ex)

    # Asynchronously schedule this function to be run again in 5 seconds
    Timer(5, greengrass_object_detection_run).start()


# Execute the function above
greengrass_object_detection_run()


# This is a dummy handler and will not be invoked
# Instead the code above will be executed in an infinite loop for our example
def function_handler(event, context):
    return

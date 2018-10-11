Minimized version from original source: https://github.com/AdityaKurude/Tensorflow-Object-Detection-API-on-AWS-Greengrass

Plus special notes when using EC2 as GGC.

1. Set up Greengrass Core - GGC v1.6.0, using Amazon EC2.
2. Install dependencies for TensorFlow on GGC:

    $ sudo pip install numpy scipy tensorflow pillow

Use *sudo* and *--no-cache-dir* option to install on EC2,  python2.7.

Package |	Local-macOS|	EC2-Linux
------------ | -------------
Numpy   | 1.15.2  |	1.15.2
Scipy	| 1.1.0   |	1.1.0
Tensorflow | 1.10.0 |	1.11.0
Pillow	| 5.3.0	| 5.3.0

3. Download `ssd_mobilenet_v1_coco` from [the TF model zoo] (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and use `frozen_inference_graph.pb`  to create zip file: `tf_models.zip`. Upload zip file to S3:
    * S3 bucket name: e.g. `qt_greengrass`
    * Note: Download the label map for the model (`models/mscoco_label_map.pbtxt`) from [TF github] (https://github.com/tensorflow/models/tree/master/research/object_detection/data)
4. [Lambda console] Create and Publish Lambda function using source from this repo:
    * Name: greengrassObjectDetection
    * Handler: greengrassObjectDetection.function_handler

5. Add Lambda function to Greengrass group
    * Memory limit:  **512MB** (observed on MacOS Activity Monitor ~325MB)
    * Timeout: 10 seconds
6. [Greengrass Group] Add Subscription: Lambda -> IoTCloud (Lambda publish message at topic "hello/world")
7. Greengrass console > Add machine learning resource
    * Model from S3: select file from step 3
    * Local path: /greengrass-machine-learning/tf (must follow what is declared in Lambda function)

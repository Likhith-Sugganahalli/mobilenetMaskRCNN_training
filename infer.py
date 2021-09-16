import os
import sys
import random
import math
import re
import datetime
import numpy as np
import tensorflow as tf
#import matplotlib
#import matplotlib.pyplot as plt
import skimage.io
# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mmrcnn import utils
from mmrcnn import visualize
from mmrcnn.visualize import display_images
import mmrcnn.model as modellib
from mmrcnn.model import log
from mmrcnn.config import Config

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "data/")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
#DEFAULT_WEIGHTS = os.path.join(ROOT_DIR, "mobile_mask_rcnn_coco.h5")

DEFAULT_WEIGHTS = os.path.join(ROOT_DIR, 'mobile_mask_rcnn_coco.h5')

# Override the training configurations with a few
# changes for inferencing.


# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_WEIGHTS_DIR = os.path.join(ROOT_DIR, "weights")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_DIR = os.path.join(ROOT_DIR, "data/coco")
DEFAULT_DATASET_YEAR = "2017" #"2014"

# Path to trained weights file
#COCO_MODEL_PATH = os.path.join(DEFAULT_WEIGHTS_DIR, "mobile_mask_rcnn_cocoperson.h5")

class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    ## Give the configuration a recognizable name
    NAME = "coco"

    ## GPU
    IMAGES_PER_GPU = 1
    GPU_COUNT = 2

    ## Number of classes (including background)
    NUM_CLASSES = 1 + 80

    ## Backbone Architecture
    BACKBONE = "mobilenetv1"
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    ## Resolution
    RES_FACTOR = 2
    IMAGE_MAX_DIM = 1024 // RES_FACTOR
    RPN_ANCHOR_SCALES = tuple(np.divide((32, 64, 128, 256, 512),RES_FACTOR))

    ## Losses
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    ## Steps
    STEPS_PER_EPOCH = 10000
    VALIDATION_STEPS = 50

    ## Additions
    TRAIN_BN = True
    POST_NMS_ROIS_INFERENCE = 100

config = CocoConfig()
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    POST_NMS_ROIS_INFERENCE = 100

config = InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
#DEVICE = "/cpu:0"
DEVICE = "/gpu:0"

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"
#TEST_MODE = "training"

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode=TEST_MODE, model_dir=MODEL_DIR,config=config)
# Set path to model weights
weights_path = DEFAULT_WEIGHTS
#weights_path = model.find_last()[1]

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)
print(model.keras_model.summary())
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

''''''
#classes = ["choco_pie_a"]

# Load random images from the images folder
NUM_IMAGES=1
images = []
file_names = next(os.walk(IMAGE_DIR))[2]
for i in range(NUM_IMAGES):
    # Read Image
    images.append(skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names))))

# Detection
times = []
for image in images:
    # Run detection
    start = datetime.datetime.now()
    results = model.detect([image], verbose=1)
    stop = datetime.datetime.now()
    t = (stop-start).total_seconds()
    times.append(t)
    # Visualize results
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],class_names, r['scores'])
    print("elapsed time for detection: {}s".format(t))

#print("median FPS: {}".format(1./np.median(times)))

print("median FPS: {}".format(1./np.median(times)))
print(np.median(times))

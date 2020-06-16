
import cv2
import os
import sys
import random
import math
import numpy as np
import skimage.io
import skimage.io.manage_plugins
import skimage.exposure
import matplotlib
import matplotlib.pyplot as plt
from tensorflow import ConfigProto
from tensorflow import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


from numpy import zeros
# Root directory of the project
ROOT_DIR = os.path.abspath("../")
# Import Mask_RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from MaskRCNN.mrcnn import utils
import MaskRCNN.mrcnn.model as modellib
from MaskRCNN.mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import MaskRCNN.samples.coco.coco as coco


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
COCO_MODEL_PATH = "/home/bernihoh/Bachelor/MaskRCNN/mask_rcnn_coco.h5"
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

#%% md

## Class Names


# Load COCO dataset
COCO_DIR = "/home/bernihoh/Bachelor/MaskRCNN/samples/coco"    # geändert: Zeile eingefügt
dataset = coco.CocoDataset()
dataset.load_coco(COCO_DIR, "train")
dataset.prepare()

# Print class names
print(dataset.class_names)

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


# Load a random image from the images folder

image = skimage.io.imread("/home/bernihoh/Bachelor/MaskRCNN/ownimages/138.jpg")

plt.imshow(image)
# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
used_class = r["class_ids"]
print(used_class)
mask = r["masks"]
mask = mask.astype(np.ubyte)
# maskimg = mask[:, :, 1] ^ mask[:, :, 1]
maskimg = np.zeros((image.shape[0], image.shape[1]))
maskimg = maskimg.astype(np.ubyte)
for i in range(mask.shape[2]):
    #skimage.io.imshow(mask[:, :, i])
    #plt.show()
    # maskimg = maskimg | mask[:, :, i]
    a = used_class[i]-1
    if used_class[i]-1 < 0:
        a = 0

    maskimg = np.maximum(maskimg,  mask[:, :, i]*a)
# maskimg[maskimg == 0] = 124
# maskimg = skimage.exposure.rescale_intensity(maskimg)
skimage.io.imshow(maskimg)
plt.show()
# skimage.io.imsave("/home/bernihoh/Bachelor/MaskRCNN/ownimages/mask138-1.jpg", maskimg)

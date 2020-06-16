
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
# ROOT_DIR = "/home/bernihoh/Bachelor/MaskRCNN/"
# Import Mask_RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/shapes/"))  # To find local version
import train_coco_person

MODEL_DIR = "/home/bernihoh/Bachelor/MaskRCNN/logsCocoPerson/"

PERSON_MODEL_PATH = "/home/bernihoh/Bachelor/MaskRCNN/logsCocoPerson/shapes20200103T2203/mask_rcnn_shapes_0002.h5"

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class InferenceConfig(train_coco_person.CocoPersonConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(PERSON_MODEL_PATH, by_name=True)

PERSON_DIR = "/home/bernihoh/Bachelor/MaskRCNN/samples/shapes"    # geändert: Zeile eingefügt
dataset = train_coco_person.CocoPersonDataset()
dataset.load_coco_person("/home/bernihoh/Bachelor/cocostuffapi/PythonAPI/pycocotools/coco/", "owntrain2017")uint8
dataset.prepare()

image = skimage.io.imread("/home/bernihoh/Bachelor/cocostuffapi/PythonAPI/pycocotools/coco/train2017/000000000831.jpg")
print("blablabla")
plt.imshow(image)
plt.show()
# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], r['scores'])
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
    maskimg = np.maximum(maskimg,  mask[:, :, i]*(used_class[i]))
# maskimg[maskimg == 0] = 124
# maskimg = skimage.exposure.rescale_intensity(maskimg)
skimage.io.imshow(maskimg)
plt.show()
print("Ende")
# skimage.io.imsave("/home/bernihoh/Bachelor/MaskRCNN/ownimages/mask138-1.jpg", maskimg)

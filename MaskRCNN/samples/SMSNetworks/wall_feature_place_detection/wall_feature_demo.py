import os
import sys
import numpy as np
import skimage.io
import skimage.exposure
import matplotlib.pyplot as plt
from tensorflow import ConfigProto
from tensorflow import InteractiveSession
from MaskRCNN.mrcnn import utils
import MaskRCNN.mrcnn.model as modellib
from MaskRCNN.mrcnn import visualize
import MaskRCNN.samples.SMSNetworks.wall_feature_place_detection.wall_feature_detection as wall_feature_detection
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from numpy import zeros
# Root directory of the project
ROOT_DIR = "/home/bernihoh/Bachelor/SMS/MaskRCNN/samples/SMSNetworks/wall_feature_place_detection/"

# Import Mask_RCNN
#sys.path.append(ROOT_DIR)  # To find local version of the library

# Import COCO config
#sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logsFaceFeatureDetection")

# Local path to trained weights file
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
COCO_MODEL_PATH = "/home/bernihoh/Bachelor/SMS/MaskRCNN/samples/SMSNetworks/wall_feature_place_detection/mask_rcnn_wall_feature_detection_0200.h5"

# Directory of images to run detection on
#IMAGE_DIR = os.path.join(ROOT_DIR, "pic_train")


class InferenceConfig(wall_feature_detection.CocoWallFeatureDetectionConfig):
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

# Load COCO dataset
# COCO_DIR = "/home/bernihoh/Bachelor/MaskRCNN/samples/coco"    # geändert: Zeile eingefügt
#dataset = head_detection.CocoHeadDetectionDataset()
#dataset.load_head_detection("/home/bernihoh/Bachelor/SMS/MaskRCNN/samples/SMSNetworks/head_place_detection/", "train")
#dataset.prepare()


class_names = ["bg", "head"]
# Load a random image from the images folder
"""
for i in range(8):
    image = skimage.io.imread("/home/bernihoh/Bachelor/SMS/MaskRCNN/samples/SMSNetworks/head_place_detection/pic_val/"+str(i)+".png")

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
        maskimg = np.maximum(maskimg,  mask[:, :, i]*(used_class[i]))
    skimage.io.imshow(maskimg)
    plt.show()
    # skimage.io.imsave("/home/bernihoh/Bachelor/MaskRCNN/ownimages/mask138-1.jpg", maskimg)
"""
image = skimage.io.imread("/home/bernihoh/Bachelor/SMS/MaskRCNN/samples/SMSNetworks/wall_feature_place_detection/pic_val/0.png")

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
    maskimg = np.maximum(maskimg,  mask[:, :, i]*(used_class[i]))
skimage.io.imshow(maskimg)
plt.show()
# skimage.io.imsave("/home/bernihoh/Bachelor/MaskRCNN/ownimages/mask138-1.jpg", maskimg)


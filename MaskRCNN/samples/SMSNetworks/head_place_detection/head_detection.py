import json
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.draw import polygon
import skimage.io as io
from PIL import Image
from tensorflow import ConfigProto
from tensorflow import InteractiveSession
from MaskRCNN.mrcnn.config import Config
from MaskRCNN.mrcnn import utils
import MaskRCNN.mrcnn.model as modellib
from MaskRCNN.mrcnn import visualize
from MaskRCNN.mrcnn.model import log
from MaskRCNN.mrcnn.config import Config
from MaskRCNN.mrcnn import utils
import MaskRCNN.mrcnn.model as modellib
from MaskRCNN.mrcnn import visualize
from MaskRCNN.mrcnn.model import log
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
ROOT_DIR = "/home/bernihoh/Bachelor/SMS/MaskRCNN/samples/SMSNetworks/head_place_detection/"
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logsHeadDetection")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join("/home/bernihoh/Bachelor/SMS/MaskRCNN/", "mask_rcnn_coco.h5")


class CocoHeadDetectionConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "head detection"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 person

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (32, 64, 90, 140, 180)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


#config = CocoHeadDetectionConfig()
#config.display()


class CocoHeadDetectionDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def __init__(self):
        super().__init__()
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}
        self.id_picname_mask_map = []

    def load_head_detection(self, base_dir, subset):
        print(base_dir)
        print(subset)
        # Add classes
        self.add_class("head_detection", 1, "head")


        # Train or validation dataset?
        assert subset in ["train", "val"]
        class_id_sequence_in_mask = []
        dataset_dir = os.path.join(base_dir, "pic_"+subset)
        dataset = os.listdir(dataset_dir)
        annotations_json_dir = os.path.join(base_dir, "pic_"+subset+".json")
        annotations = json.load(open(annotations_json_dir))["_via_img_metadata"]
        for annotation, counter in zip(annotations, range(0, len(annotations))):
            picture_annotation = annotations[annotation]
            picture = io.imread(os.path.join(dataset_dir, picture_annotation["filename"]))
            width_pic, height_pic = picture.shape[1], picture.shape[0]
            mask = np.zeros([height_pic, width_pic, len(picture_annotation["regions"])], dtype=np.uint8)
            for region, i in zip(picture_annotation["regions"], range(len(picture_annotation["regions"]))):
                rr, cc = polygon(region["shape_attributes"]["all_points_y"], region["shape_attributes"]["all_points_x"])
                try:
                    mask[rr, cc, i] = 1
                except:
                    pass
            bool_mask = mask.astype(np.bool)
            self.id_picname_mask_map.append({"id": counter, "picname": picture, "mask": mask, "bool_mask": bool_mask, "class_id_sequence": [1]})
            self.add_image("head_detection", image_id=counter, path=os.path.join(dataset_dir, picture_annotation["filename"]), picname=picture_annotation["filename"])

    def image_reference(self, image_id):
        """Return the data of the image."""
        return self.image_info[image_id]

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        return_mask = self.id_picname_mask_map[image_id]["bool_mask"]
        return_class_id_sequence = self.id_picname_mask_map[image_id]["class_id_sequence"]
        return return_mask, np.ones([return_mask.shape[-1]], dtype=np.uint32)
        # return return_mask, return_class_id_sequence
"""
# Training dataset
dataset_train = CocoHeadDetectionDataset()
dataset_train.load_head_detection("/home/bernihoh/Bachelor/SMS/MaskRCNN/samples/SMSNetworks/head_place_detection/", "train")
dataset_train.prepare()

# Validation dataset
dataset_val = CocoHeadDetectionDataset()
# dataset_val.load_head_detection("/home/bernihoh/Bachelor/SMS/MaskRCNN/samples/SMSNetworks/head_place_detection/", "val")
#!!!!!!!!!Change in the line below "train" to "val" when finished with testing!!!!!!!!!!!!!!!!!!!!!!!!
dataset_val.load_head_detection("/home/bernihoh/Bachelor/SMS/MaskRCNN/samples/SMSNetworks/head_place_detection/", "val")
dataset_val.prepare()

# Create Model
# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

# Which weights to start with? init_with = "coco"  other options: (imagenet, coco, or last or own)
init_with = "own"
if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":  # Load weights trained on MS COCO, but skip layers that are different due to the different number of classes. See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":  # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)
elif init_with == "own":
    model.load_weights("/home/bernihoh/Bachelor/SMS/MaskRCNN/samples/SMSNetworks/head_place_detection/mask_rcnn_head detection_0020.h5")

model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=1, layers='heads')
model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10, epochs=20, layers="all")


# Save weights, Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)

# Detection

class InferenceConfig(CocoHeadDetectionConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
# Get path to saved weights, Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

counter = 0

# Test on a random image
while True:
    counter += 1
    #image_id = random.choice(dataset_val.image_ids)
    image_id = 7
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, inference_config, image_id, use_mini_mask=False)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)
    try:
        visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, dataset_train.class_names, figsize=(8, 8))
        results = model.detect([original_image], verbose=1)
        r = results[0]
        visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], dataset_val.class_names, r['scores'])
        # visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], dataset_val.class_names, r['scores'], ax=get_ax())
        print(counter)
        break
    except:
        pass
"""
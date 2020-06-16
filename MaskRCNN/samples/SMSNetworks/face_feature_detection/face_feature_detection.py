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
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
ROOT_DIR = "/home/bernihoh/Bachelor/SMS/MaskRCNN/samples/SMSNetworks/face_feature_detection/"
sys.path.append(ROOT_DIR)  # To find local version of the library
MODEL_DIR = os.path.join(ROOT_DIR, "logsFaceFeatureDetection")
COCO_MODEL_PATH = os.path.join("/home/bernihoh/Bachelor/SMS/MaskRCNN/", "mask_rcnn_coco.h5")


class CocoFaceFeatureDetectionConfig(Config):
    """Configuration for training on the toy shapes dataset.
        Derives from the base Config class and overrides values specific
        to the toy shapes dataset.
        """
    # Give the configuration a recognizable name
    NAME = "face_feature_detection"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 17  # background + iris_l + iris_r + inner_eye_l + inner_eye_r + outer_eye_l + outer_eye_r +
                          # eye_brow_l + eye_brow_r + nose_tip + nose + cheek_l + cheek_r + mouth + chin + face + head + distortion

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 64
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 64% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 33

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 239

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 10


#config = CocoFaceFeatureDetectionConfig()
#config.display()


class CocoFaceFeatureDetectionDataset(utils.Dataset):
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
        self.add_class("face_feature_detection", 1, "iris_l")
        self.add_class("face_feature_detection", 2, "inner_eye_l")
        self.add_class("face_feature_detection", 3, "outer_eye_l")
        self.add_class("face_feature_detection", 4, "eye_brow_l")
        self.add_class("face_feature_detection", 5, "cheek_l")
        self.add_class("face_feature_detection", 6, "iris_r")
        self.add_class("face_feature_detection", 7, "inner_eye_r")
        self.add_class("face_feature_detection", 8, "outer_eye_r")
        self.add_class("face_feature_detection", 9, "eye_brow_r")
        self.add_class("face_feature_detection", 10, "cheek_r")
        self.add_class("face_feature_detection", 11, "nose_tip")
        self.add_class("face_feature_detection", 12, "nose")
        self.add_class("face_feature_detection", 13, "mouth")
        self.add_class("face_feature_detection", 14, "chin")
        self.add_class("face_feature_detection", 15, "face")
        self.add_class("face_feature_detection", 16, "head")
        self.add_class("face_feature_detection", 17, "distortion")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        class_id_sequence_in_mask = []
        dataset_dir = os.path.join(base_dir, "pic_"+subset)
        dataset = os.listdir(dataset_dir)
        annotations_json_dir = os.path.join(base_dir, "pic_"+subset+"_annotations.json")
        annotations = json.load(open(annotations_json_dir))

        for counter in range(0, len(annotations)):
            picture_annotation = annotations[str(counter)]
            [height_pic, width_pic, class_count] = picture_annotation["pic_shape"]
            mask = np.zeros([height_pic, width_pic, class_count], dtype=np.uint8)
            class_id_sequence = np.zeros(class_count, dtype=np.uint32)
            for i in range(0, class_count):
                class_id_name = picture_annotation["regions"][i]["annotation_class"]
                class_id_sequence[i] = (list(filter(lambda x: x["name"] == picture_annotation["regions"][i]["annotation_class"], self.class_info))[0]["id"])
                #print(class_id_sequence)
                rr, cc = polygon(picture_annotation["regions"][i]["annotation_polygon"]["all_points_y"], picture_annotation["regions"][i]["annotation_polygon"]["all_points_x"])
                try:
                    mask[rr, cc, i] = 1
                except:
                    pass
            bool_mask = mask.astype(np.bool)
            self.id_picname_mask_map.append({"id": counter, "picname": picture_annotation["pic_file_name"], "mask": mask, "bool_mask": bool_mask, "class_id_sequence": class_id_sequence})
            self.add_image("face_feature_detection", image_id=counter, path=os.path.join(dataset_dir, picture_annotation["pic_file_name"]), picname=picture_annotation["pic_file_name"])
    def image_reference(self, image_id):
        """Return the data of the image."""
        return self.image_info[image_id]

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        return_mask = self.id_picname_mask_map[image_id]["bool_mask"]
        return_class_id_sequence = self.id_picname_mask_map[image_id]["class_id_sequence"]
        # return return_mask, np.ones([return_mask.shape[-1]], dtype=np.uint32)
        return return_mask, return_class_id_sequence #.astype(np.uint32)
"""
# Training dataset
dataset_train = CocoFaceFeatureDetectionDataset()
dataset_train.load_head_detection("/home/bernihoh/Bachelor/SMS/MaskRCNN/samples/SMSNetworks/face_feature_detection/", "train")
dataset_train.prepare()

# Validation dataset
dataset_val = CocoFaceFeatureDetectionDataset()
# dataset_val.load_head_detection("/home/bernihoh/Bachelor/SMS/MaskRCNN/samples/SMSNetworks/head_place_detection/", "val")
#!!!!!!!!!Change in the line below "train" to "val" when finished with testing!!!!!!!!!!!!!!!!!!!!!!!!
dataset_val.load_head_detection("/home/bernihoh/Bachelor/SMS/MaskRCNN/samples/SMSNetworks/face_feature_detection/", "val")
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
    model.load_weights("/home/bernihoh/Bachelor/SMS/MaskRCNN/samples/SMSNetworks/face_feature_detection/mask_rcnn_face_feature_detection_0029.h5")

model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=1, layers='heads')
model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10, epochs=100, layers="all")

# Save weights, Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)

# Detection

class InferenceConfig(CocoFaceFeatureDetectionConfig):
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
    image_id = 150
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

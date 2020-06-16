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
import skimage.draw
import skimage.io
from PIL import Image
from tensorflow import ConfigProto
from tensorflow import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Root directory of the project
# ROOT_DIR = os.path.abspath("../../")
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logsCocoPerson")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


class CocoPersonConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 person

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


config = CocoPersonConfig()
config.display()

# Notebook Preferences


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


class CocoPersonTreeDataset(utils.Dataset):
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

    def load_coco_person(self, base_dir, subset):
        print(base_dir)
        print(subset)
        # Add classes
        self.add_class("coco_person_tree", 1, "person")
        self.add_class("coco_person_tree", 169, "tree")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        class_id_sequence_in_mask = []
        dataset_dir = os.path.join(base_dir, subset)
        dataset = os.listdir(dataset_dir)
        # dataset = sorted(os.listdir(dataset_dir))
        # print(dataset)

        annotations_json_dir = os.path.join(base_dir, "concatenated/"+subset+"annotations.json")
        annotations_load = json.load(open(annotations_json_dir))
        # annotations_values = list(annotations_load.values())
        annotations = []
        for a in annotations_load:  # instead of annotations_values
            annotations_item = {"image_id": a["image_id"], "bbox": a["bbox"], "category_id": a["category_id"]}
            annotations.append(annotations_item)
        border_table_dir = os.path.join(base_dir, "concatenated/" + subset + "border_table.json")
        border_table = json.load(open(border_table_dir))
        counter = 0
        for picture in dataset:
            print("counter", counter)
            picture_name = picture[0:-4]
            print(picture, picture_name)
            image_id = int(picture_name.replace("0", ""))
            annotations_start = -1
            annotations_end = -1
            for a in border_table:
                if a[0]["image_id"] == image_id:
                    print("border", a[0])
                    annotations_start = a[0]["start_image_id_pos"]
                    annotations_end = a[0]["end_image_id_pos"]
            if annotations_start > -1 and annotations_end > -1:

                img = cv2.imread(os.path.join(dataset_dir, picture), 0)  # 0 for grey scale
                width_img, height_img = img.shape
                # img = int(img)
                mask = np.zeros([height_img, width_img, (annotations_end - annotations_start)], dtype=np.uint8)
                counter2 = 0
                for b in annotations[annotations_start: annotations_end]:
                    category_id = b["category_id"]
                    if category_id == 1 or category_id == 169:
                        bbox = b["bbox"]
                        bbox_height = int(bbox[3])
                        bbox_width = int(bbox[2])
                        x1 = int(bbox[0])
                        y1 = int(bbox[1])
                        x2 = x1 + bbox_width
                        y2 = y1 + bbox_height
                        print(bbox_height, bbox_width)
                        print("image_shape", img.shape)
                        bbox_in_img = img[x1:x2, y1:y2]
                        print(bbox_in_img.shape)
                        used = 0
                        for i in range(bbox_width):
                            for j in range(bbox_height):
                                try:
                                    if bbox_in_img[i][j] == category_id:
                                        mask[i][j][counter2] = category_id
                                        used = 1
                                except:
                                    pass
                        if used == 1:
                            class_id_sequence_in_mask.append(category_id)
                    counter2 += 1
                bool_mask = mask.astype(np.bool)
                self.id_picname_mask_map.append({"id": counter,
                                                 "picname": picture,
                                                 "mask": mask,
                                                 "bool_mask": bool_mask,
                                                 "class_id_sequence": class_id_sequence_in_mask})
                print("len", len(self.id_picname_mask_map))

                self.add_image("coco_person_tree",
                               image_id=(len(self.id_picname_mask_map) - 1 ),
                               path=os.path.join(dataset_dir, picture),
                               picname=picture)


        """
        for a in annotations:
            if type(a["regions"]) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
            polygon_id_map_item = {"id": annotation_counter, "polygons": polygons}
            polygon_id_map.append(polygon_id_map_item)
            annotation_counter += 1
        for i in range(annotation_counter):
            mask_info = polygon_id_map[i]["polygons"]
            # print(mask_info)
            file_in_dataset_dir = os.path.join(dataset_dir, dataset[i])
            image = Image.open(file_in_dataset_dir, mode="r")
            # print(dataset[i])
            width, height = image.size
            # print(width, height)

            mask = np.zeros([height, width, len(mask_info)],
                            dtype=np.uint8)
            picture_mask = np.zeros([height, width],
                                    dtype=np.uint8)
            for j, p in enumerate(mask_info):
                print("ä")
                xx, yy = skimage.draw.polygon(p["all_points_y"], p["all_points_x"])
                mask[xx, yy, j] = 1
                picture_mask[xx, yy] = 100

            #img = skimage.img_as_ubyte(picture_mask)
            #skimage.io.imshow(img)
            #plt.show()
            bool_mask = mask.astype(np.bool)
            self.id_picname_mask_map.append({"id": i,
                                                "picname": dataset[i],
                                                "mask": mask,
                                                "bool_mask": bool_mask})
            # print(self.id_picname_mask_map[i])

            self.add_image("coco_person",
                            image_id=i,
                            path=file_in_dataset_dir,
                            picname=dataset[i],
                            )
            """


        # for i in range(len(dataset)):
        #     self.add_image("coco_person",
        #                    image_id=i,
        #                    path=os.path.join(dataset_dir, dataset[i]),
        #                    )

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


# Training dataset
dataset_train = CocoPersonTreeDataset()
dataset_train.load_coco_person("/home/bernihoh/Bachelor/cocostuffapi/PythonAPI/pycocotools/person_tree/",
                               "train")
dataset_train.prepare()

# Validation dataset
dataset_val = CocoPersonTreeDataset()
dataset_val.load_coco_person("/home/bernihoh/Bachelor/cocostuffapi/PythonAPI/pycocotools/person_tree/",
                             "val")
dataset_val.prepare()


# Create Model
# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# %%

# Which weights to start with?
# init_with = "coco"  # imagenet, coco, or last or own
init_with = "last"
if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)
elif init_with == "own":
    model.load_weights("/home/bernihoh/Bachelor/MaskRCNN/logsCocoPerson/shapes20200103T2203/mask_rcnn_shapes_0002.h5")


model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=1, layers='heads')
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=2,
            layers="all")


# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)


# Detection

class InferenceConfig(CocoPersonConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)
# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Test on a random image
image_id = random.choice(dataset_val.image_ids)

original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
    modellib.load_image_gt(dataset_val, inference_config,
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

# visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
  #                           dataset_train.class_names, figsize=(8, 8))

results = model.detect([original_image], verbose=1)

r = results[0]
# visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
  #                           dataset_val.class_names, r['scores'], ax=get_ax())

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
from numba import cuda
import MaskRCNN.samples.SMSNetworks.face_feature_detection.face_feature_detection as face_feature_detection


class InferenceConfig(face_feature_detection.CocoFaceFeatureDetectionConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)


class face_feature_detection:
    def detect(self, image):
        cuda.select_device(0)

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        ROOT_DIR = "/home/bernihoh/Bachelor/SMS/MaskRCNN/samples/SMSNetworks/face_feature_detection/"
        MODEL_DIR = os.path.join(ROOT_DIR, "logsFaceFeatureDetection")
        COCO_MODEL_PATH = "/home/bernihoh/Bachelor/SMS/MaskRCNN/samples/SMSNetworks/face_feature_detection/mask_rcnn_face_feature_detection_0029.h5"
        config = InferenceConfig()
        config.display()

        # Create model object in inference mode.
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        model.load_weights(COCO_MODEL_PATH, by_name=True)

        class_names = ["bg", "iris_l", "inner_eye_l", "outer_eye_l", "eye_brow_l", "cheek_l", "iris_r",
                       "inner_eye_r", "outer_eye_r", "eye_brow_r", "cheek_r", "nose_tip", "nose", "mouth",
                       "chin", "face", "head", "distortion"]

        results = model.detect([image], verbose=1)
        r = results[0]
        session.close()
        cuda.close()
        return r


if __name__ == "__main__":
    face_feature_detection.detect(face_feature_detection(), skimage.io.imread("/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/face_net_pics/1.png"))

from abc import *
import cv2
import os
import sys
import random
import math
import numpy as np
import skimage.io
import skimage.io.manage_plugins
import skimage.exposure
import pickle as pkl
import random
import matplotlib
import matplotlib.pyplot as plt
from tensorflow import ConfigProto
from tensorflow import InteractiveSession
from numba import cuda
from MaskRCNN.mrcnn import utils
import MaskRCNN.mrcnn.model as modellib
from MaskRCNN.mrcnn import visualize
import MaskRCNN.samples.coco.coco as coco
from modification.helper_files import roi_helper
from modification.netcompare.services.face_net import face_net_manager_old, face_net_manager
import MaskRCNN.samples.SMSNetworks.face_feature_detection.face_feature_detection as face_feature_detection
import MaskRCNN.samples.SMSNetworks.head_place_detection.head_detection as head_detection
import MaskRCNN.samples.SMSNetworks.wall_feature_place_detection.wall_feature_detection as wall_feature_detection


import array


class InferenceConfigFFN(face_feature_detection.CocoFaceFeatureDetectionConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)


class InferenceConfigHDN(head_detection.CocoHeadDetectionConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    RPN_ANCHOR_SCALES = (32, 64, 90, 140, 180)


class InferenceConfigWFN(wall_feature_detection.CocoWallFeatureDetectionConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    RPN_ANCHOR_SCALES = (64, 80, 128, 160, 256)


class detection(ABC):

    def __init__(self, image):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        self.ROOT_DIR = os.path.abspath("/home/bernihoh/Bachelor/SMS/MaskRCNN/samples/")
        self.model = None
        self.InferenceConfig = None
        self.image = image
        self.MODEL_DIR = ""
        self.NETWORK_PATH = ""

    # template method pattern
    def detect(self):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        self.MODEL_DIR = self.model_dir()
        self.NETWORK_PATH = self.network_path()
        self.InferenceConfig = self.inference_config()
        self.InferenceConfig.display()
        self.model = modellib.MaskRCNN(mode="inference", model_dir=self.MODEL_DIR, config=self.InferenceConfig)
        self.model.load_weights(self.NETWORK_PATH, by_name=True)
        results = self.model.detect([self.image], verbose=1)
        """r looks like this: r['rois'], r['masks'], r['class_ids'], r['scores']"""
        r = results[0]

        class_names = self.class_names()

        visualize.display_instances(self.image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        session.close()
        ret = self.further_image_manipulation(r)
        return ret

    # abstract method
    def class_names(self):
        raise Exception("You are using the abstract class 'detection'")

    # abstract method
    def inference_config(self):
        raise Exception("You are using the abstract class 'detection'")

    # abstract method
    def model_dir(self):
        raise Exception("You are using the abstract class 'detection'")

    # abstract method
    def network_path(self):
        raise Exception("You are using the abstract class 'detection'")

    # abstract method
    def further_image_manipulation(self, r):
        raise Exception("You are using the abstract class 'detection'")


class FFN(detection):

    def inference_config(self):
        InferenceConfig = InferenceConfigFFN()
        InferenceConfig.NAME = "face_feature_detection"
        InferenceConfig.GPU_COUNT = 1
        InferenceConfig.IMAGES_PER_GPU = 1
        InferenceConfig.NUM_CLASSES = 1 + 17  # background + iris_l + iris_r + inner_eye_l + inner_eye_r + outer_eye_l + outer_eye_r +
                                                   # eye_brow_l + eye_brow_r + nose_tip + nose + cheek_l + cheek_r + mouth + chin + face + head + distortion
        InferenceConfig.IMAGE_MIN_DIM = 64
        InferenceConfig.IMAGE_MAX_DIM = 512
        InferenceConfig.RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
        return InferenceConfig

    def class_names(self):
        return ["bg", "iris_l", "inner_eye_l", "outer_eye_l", "eye_brow_l", "cheek_l", "iris_r",
                       "inner_eye_r", "outer_eye_r", "eye_brow_r", "cheek_r", "nose_tip", "nose", "mouth",
                       "chin", "face", "head", "distortion"]

    """@return: the Path to the log-directory"""
    def model_dir(self):
        return os.path.join(self.ROOT_DIR, "SMSNetworks/face_feature_detection/" + "logsFaceFeatureDetection")

    """@return: the Path to the .h5 file that contains the Face Feature Network"""
    def network_path(self):
        return os.path.join(self.ROOT_DIR, "SMSNetworks/face_feature_detection/"+"mask_rcnn_face_feature_detection_0029.h5")

    """The classes of FFN are: iris_l/r, inner_eye_l/r, outer_eye_l/r, eye_brow_l/r, cheek_l/r, nose_tip, nose, mouth, chin,
        head, face and distortion. This method detects to which side (left or right) these features belong
        and it calculates crucial points like the iris_center, eye_bridge, 3 nose_tip points, 5 eye_brow 
        points per eye brow, 3 mouth points, 3 chin points, 1 cheek_points per cheek, 2ear_attachment points
        and 2 eye_meridian_points. 
        This method returns the following feature_ids: iris_center_l/r, iris_area_l/r, inner_eye_area_l/r, 
        inner_eye_point_array_l/r, outer_eye_point_array_l/r, outer_eye_area_l/r, cheek_point_array_l/r, 
        eye_brow_point_array_l/r, eye_bridge_point, nose_tip_point_array, mouth_area, mouth_point_array, 
        chin_point_array, ear_attachment_l/r, eye_meridian_l/r.
        @return: (image, [feature_ids], hash map: {feature_id: [feature_mask/point/pointarray, roi]}, face_mask, face_roi, head_mask, head_roi)"""
    def further_image_manipulation(self, r):
        class_names = {"bg": 0, "iris_l": 1, "inner_eye_l": 2, "outer_eye_l": 3, "eye_brow_l": 4, "cheek_l": 5, "iris_r": 6,
                       "inner_eye_r": 7, "outer_eye_r": 8, "eye_brow_r": 9, "cheek_r": 10, "nose_tip": 11, "nose": 12, "mouth": 13,
                       "chin": 14, "face": 15, "head": 16, "distortion": 17}
        for class_id in ["iris_l", "inner_eye_l", "outer_eye_l", "eye_brow_l", "cheek_l", "iris_r",
                         "inner_eye_r", "outer_eye_r", "eye_brow_r", "cheek_r", "nose_tip", "nose", "mouth",
                         "chin", "face", "head"]:
            found = False
            for used_class_id in r["class_ids"]:
                if class_names[class_id] == used_class_id:
                    found = True
                    break
            if not found:
                return "redo"
        #for i in range(len(class_names)):
        #    try:
        #        plt.imshow(r["masks"][:, :, i])
        #        plt.show()
        #    except:
        #        pass
        for class_name in class_names:
            class_index_score_list = []
            for i in range(len(r["class_ids"])):
                if class_names[class_name] == r["class_ids"][i]:
                    class_index_score_list.append([i, r["scores"][i]])
            maximum_index = -1
            maximum_score = -1
            for index_score_item in class_index_score_list:
                if index_score_item[1] > maximum_score:
                    maximum_index = index_score_item[0]
                    maximum_score = index_score_item[1]
            class_names.update({class_name: maximum_index})
        """
        for class_name in class_names:
            plt.imshow(r["masks"][:, :, class_names[class_name]])
            plt.title(class_name)
            plt.show()
        
        plt.imshow(r["masks"][:, :, class_names["eye_brow_r"]])
        plt.title("eye_brow_r")
        plt.show()
        """
        #print(class_names)
        # print("iris")
        iris_area_l = roi_helper.cut_out_roi(r["masks"][:, :, class_names["iris_l"]], r["rois"][class_names["iris_l"]])
        iris_area_r = roi_helper.cut_out_roi(r["masks"][:, :, class_names["iris_r"]], r["rois"][class_names["iris_r"]])
        iris_center_l = [a+b for a, b in zip(roi_helper.find_center(iris_area_l), r["rois"][class_names["iris_l"]][0: 2])]
        iris_center_r = [a+b for a, b in zip(roi_helper.find_center(iris_area_r),  r["rois"][class_names["iris_r"]][0: 2])]
        # print("inner_eye")
        inner_eye_area_l = roi_helper.cut_out_roi(r["masks"][:, :, class_names["inner_eye_l"]], r["rois"][class_names["inner_eye_l"]])
        inner_eye_area_r = roi_helper.cut_out_roi(r["masks"][:, :, class_names["inner_eye_r"]], r["rois"][class_names["inner_eye_r"]])
        inner_eye_point_array_l = [[a+b for a, b in zip(roi_helper.find_side_centric_point(inner_eye_area_l, "left"), r["rois"][class_names["inner_eye_l"]][0: 2])], [a+b for a, b in zip(roi_helper.find_side_centric_point(inner_eye_area_l, "right"), r["rois"][class_names["inner_eye_l"]][0: 2])]]
        inner_eye_point_array_r = [[a+b for a, b in zip(roi_helper.find_side_centric_point(inner_eye_area_r, "left"), r["rois"][class_names["inner_eye_r"]][0: 2])], [a+b for a, b in zip(roi_helper.find_side_centric_point(inner_eye_area_r, "right"), r["rois"][class_names["inner_eye_r"]][0: 2])]]
        # print("outer_eye")
        outer_eye_area_l = roi_helper.cut_out_roi(r["masks"][:, :, class_names["outer_eye_l"]], r["rois"][class_names["outer_eye_l"]])
        outer_eye_area_r = roi_helper.cut_out_roi(r["masks"][:, :, class_names["outer_eye_r"]], r["rois"][class_names["outer_eye_r"]])
        outer_eye_point_array_l = [[a+b for a, b in zip(roi_helper.find_side_centric_point(outer_eye_area_l, "left"), r["rois"][class_names["outer_eye_l"]][0: 2])], [a+b for a, b in zip(roi_helper.find_side_centric_point(outer_eye_area_l, "right"), r["rois"][class_names["outer_eye_l"]][0: 2])]]
        outer_eye_point_array_r = [[a+b for a, b in zip(roi_helper.find_side_centric_point(outer_eye_area_r, "left"), r["rois"][class_names["outer_eye_r"]][0: 2])], [a+b for a, b in zip(roi_helper.find_side_centric_point(outer_eye_area_r, "right"), r["rois"][class_names["outer_eye_r"]][0: 2])]]
        # print("cheek_l")
        cheek_point_array_l = [[a+b for a, b in zip(roi_helper.find_side_centric_point(roi_helper.cut_out_roi(r["masks"][:, :, class_names["cheek_l"]], r["rois"][class_names["cheek_l"]]), "left"), r["rois"][class_names["cheek_l"]][0: 2])], [a+b for a, b in zip(roi_helper.find_side_centric_point(roi_helper.cut_out_roi(r["masks"][:, :, class_names["cheek_l"]], r["rois"][class_names["cheek_l"]]), "down"), r["rois"][class_names["cheek_l"]][0: 2])]]
        # print("cheek_r")
        cheek_point_array_r = [[a+b for a, b in zip(roi_helper.find_side_centric_point(roi_helper.cut_out_roi(r["masks"][:, :, class_names["cheek_r"]], r["rois"][class_names["cheek_r"]]), "down"), r["rois"][class_names["cheek_r"]][0: 2])], [a+b for a, b in zip(roi_helper.find_side_centric_point(roi_helper.cut_out_roi(r["masks"][:, :, class_names["cheek_r"]], r["rois"][class_names["cheek_r"]]), "right"), r["rois"][class_names["cheek_r"]][0: 2])]]
        eye_brow_point_array_l = [[x[0]+r["rois"][class_names["eye_brow_l"]][0], x[1]+r["rois"][class_names["eye_brow_l"]][1]] for x in roi_helper.find_points_on_gravity_line(roi_helper.cut_out_roi(r["masks"][:, :, class_names["eye_brow_l"]], r["rois"][class_names["eye_brow_l"]]), ["left", "right"], 5)]
        # print(eye_brow_point_array_l)
        eye_brow_point_array_r = [[x[0]+r["rois"][class_names["eye_brow_r"]][0], x[1]+r["rois"][class_names["eye_brow_r"]][1]] for x in roi_helper.find_points_on_gravity_line(roi_helper.cut_out_roi(r["masks"][:, :, class_names["eye_brow_r"]], r["rois"][class_names["eye_brow_r"]]), ["left", "right"], 5)]
        #print(eye_brow_point_array_r)
        eye_bridge_point = [int((inner_eye_point_array_l[-1][0] + inner_eye_point_array_r[0][0]) // 2), int((inner_eye_point_array_l[-1][1] + inner_eye_point_array_r[0][1]) // 2)]
        #print("nose")
        nose_area = roi_helper.cut_out_roi(r["masks"][:, :, class_names["nose"]], r["rois"][class_names["nose"]])
        nose_tip_area = roi_helper.cut_out_roi(r["masks"][:, :, class_names["nose_tip"]], r["rois"][class_names["nose_tip"]])
        nose_tip_point_array = [[a+b for a, b in zip(roi_helper.find_side_centric_point(nose_area, "left"), r["rois"][class_names["nose"]][0: 2])], [a+b for a, b in zip(roi_helper.find_cog(nose_tip_area), r["rois"][class_names["nose_tip"]][0: 2])], [a+b for a, b in zip(roi_helper.find_side_centric_point(nose_area, "right"), r["rois"][class_names["nose"]][0: 2])]]
        #print(nose_tip_point_array)
        #print("mouth")
        mouth_area = roi_helper.cut_out_roi(r["masks"][:, :, class_names["mouth"]], r["rois"][class_names["mouth"]])
        #mouth_point_array = [[a+b for a, b in zip(roi_helper.find_side_centric_point(mouth_area, "left"), r["rois"][class_names["mouth"]][0: 2])], [a+b for a, b in zip(roi_helper.find_cog(mouth_area), r["rois"][class_names["mouth"]][0: 2])], [a+b for a, b in zip(roi_helper.find_side_centric_point(mouth_area, "right"), r["rois"][class_names["mouth"]][0: 2])]]
        mouth_point_array = [[a+b for a, b in zip(roi_helper.find_side_centric_point(mouth_area, "left"), r["rois"][class_names["mouth"]][0: 2])], [int((a+b)//2) for a, b in zip([a+b for a, b in zip(roi_helper.find_side_centric_point(mouth_area, "up"), r["rois"][class_names["mouth"]][0: 2])], [a+b for a, b in zip(roi_helper.find_side_centric_point(mouth_area, "down"), r["rois"][class_names["mouth"]][0: 2])])], [a+b for a, b in zip(roi_helper.find_side_centric_point(mouth_area, "right"), r["rois"][class_names["mouth"]][0: 2])]]
        #print(mouth_point_array)
        #print("chin")
        chin_area = roi_helper.cut_out_roi(r["masks"][:, :, class_names["chin"]], r["rois"][class_names["chin"]])
        chin_point_array = [[a+b for a, b in zip(roi_helper.find_side_centric_point(chin_area, "left"), r["rois"][class_names["chin"]][0: 2])], [a+b for a, b in zip(roi_helper.find_side_centric_point(chin_area, "up"), r["rois"][class_names["chin"]][0: 2])], [a+b for a, b in zip(roi_helper.find_side_centric_point(chin_area, "right"), r["rois"][class_names["chin"]][0: 2])]]
        #print("face")
        full_face_area = np.array(np.bitwise_or(r["masks"][:, :, class_names["face"]], r["masks"][:, :, class_names["chin"]]), dtype=np.uint8)
        #print("head")
        full_head_area = np.array(np.bitwise_or(r["masks"][:, :, class_names["head"]], full_face_area), dtype=np.uint8)
        ear_meridian = int((roi_helper.find_side_centric_point(full_head_area, "up")[0] + roi_helper.find_side_centric_point(full_head_area, "down")[0])*1.2 // 2)
        #print(ear_meridian, "ear_meridian")
        ear_meridian_l, ear_meridian_r = roi_helper.find_most_outer_points_of_array(full_head_area[ear_meridian])
        ear_attachment_l = [ear_meridian, ear_meridian_l]
        ear_attachment_r = [ear_meridian, ear_meridian_r]
        #print(ear_attachment_l, ear_attachment_r)
        eye_meridian_l, eye_meridian_r = roi_helper.find_most_outer_points_of_array(full_face_area[eye_bridge_point[0]])
        eye_meridian_l = [eye_bridge_point[0], eye_meridian_l]
        eye_meridian_r = [eye_bridge_point[0], eye_meridian_r]
        #print(eye_meridian_l, eye_meridian_r)
        ret = {"image": self.image,
               "feature_ids": ["iris_area_l", "iris_area_r", "iris_center_l", "iris_center_r", "inner_eye_area_l", "inner_eye_area_r",
                               "inner_eye_point_array_l", "inner_eye_point_array_r", "inner_eye_point_array_l", "inner_eye_point_array_r",
                               "outer_eye_area_l", "outer_eye_area_r", "outer_eye_point_array_l", "outer_eye_point_array_r",
                               "cheek_point_array_l", "cheek_point_array_r", "eye_brow_point_array_l", "eye_brow_point_array_r",
                               "eye_bridge_point", "nose_tip_point_array", "mouth_area", "mouth_point_array",
                               "chin_point_array", "ear_attachment_l", "ear_attachment_r", "eye_meridian_l", "eye_meridian_r"],
               "face_net": {"iris_area_l": iris_area_l, "iris_area_r": iris_area_r, "iris_center_point_l": iris_center_l, "iris_center_point_r": iris_center_r,
                            "inner_eye_area_l": inner_eye_area_l, "inner_eye_area_r": inner_eye_area_r, "inner_eye_point_array_l": inner_eye_point_array_l, "inner_eye_point_array_r": inner_eye_point_array_r,
                            "outer_eye_area_l": outer_eye_area_l, "outer_eye_area_r": outer_eye_area_r, "outer_eye_point_array_l": outer_eye_point_array_l, "outer_eye_point_array_r": outer_eye_point_array_r,
                            "cheek_point_array_l": cheek_point_array_l, "cheek_point_array_r": cheek_point_array_r,
                            "eye_brow_point_array_l": eye_brow_point_array_l, "eye_brow_point_array_r": eye_brow_point_array_r,
                            "eye_bridge_point": eye_bridge_point, "nose_tip_point_array": nose_tip_point_array,
                            "mouth_area": mouth_area, "mouth_point_array": mouth_point_array,
                            "chin_point_array": chin_point_array,
                            "ear_attachment_l": ear_attachment_l, "ear_attachment_r": ear_attachment_r,
                            "eye_meridian_l": eye_meridian_l, "eye_meridian_r": eye_meridian_r},
               "face_score_input": {"points": {"iris_center_point_l": iris_center_l, "iris_center_point_r": iris_center_r, "inner_eye_point_array_l": inner_eye_point_array_l, "inner_eye_point_array_r": inner_eye_point_array_r, "outer_eye_point_array_l": outer_eye_point_array_l, "outer_eye_point_array_r": outer_eye_point_array_r, "cheek_point_array_l": cheek_point_array_l, "cheek_point_array_r": cheek_point_array_r, "eye_brow_point_array_l": eye_brow_point_array_l, "eye_brow_point_array_r": eye_brow_point_array_r, "eye_bridge_point": eye_bridge_point, "nose_tip_point_array": nose_tip_point_array, "mouth_point_array": mouth_point_array, "chin_point_array": chin_point_array},
                                    "areas": {"iris_area_l": iris_area_l, "iris_area_r": iris_area_r, "inner_eye_area_l": inner_eye_area_l, "inner_eye_area_r": inner_eye_area_r, "outer_eye_area_l": outer_eye_area_l, "outer_eye_area_r": outer_eye_area_r, "mouth_area": mouth_area}
                                    },
               "face_mask": full_face_area,
               "face_roi": [min(r["rois"][class_names["face"]][0], r["rois"][class_names["chin"]][0]), min(r["rois"][class_names["face"]][1], r["rois"][class_names["chin"]][1]), max(r["rois"][class_names["face"]][2], r["rois"][class_names["chin"]][2]), max(r["rois"][class_names["face"]][3], r["rois"][class_names["chin"]][3])],
               "head_mask": full_head_area,
               "head_roi": [min(r["rois"][class_names["face"]][0], r["rois"][class_names["head"]][0], r["rois"][class_names["chin"]][0]), min(r["rois"][class_names["face"]][1], r["rois"][class_names["head"]][1], r["rois"][class_names["chin"]][1]), max(r["rois"][class_names["face"]][2], r["rois"][class_names["head"]][2], r["rois"][class_names["chin"]][3]), max(r["rois"][class_names["face"]][3], r["rois"][class_names["head"]][3], r["rois"][class_names["chin"]][3])]}
        return ret


class HDN(detection):

    def inference_config(self):
        InferenceConfig = InferenceConfigHDN()
        return InferenceConfig

    def class_names(self):
        return ["bg", "head"]

    """@return: the Path to the log-directory"""
    def model_dir(self):
        return os.path.join(self.ROOT_DIR, "SMSNetworks/head_place_detection/" + "logsHeadDetection")

    """@return: the Path to the .h5 file that contains the Head Detection Network"""
    def network_path(self):
        return os.path.join(self.ROOT_DIR, "SMSNetworks/head_place_detection/"+"mask_rcnn_head detection_0020.h5")

    """The classes of HDN are: bg, head. This method returns the id, the masks of the head places, the rois and the 
            ratio a head covers proportional to the whole picture
            @return: {id: [mask, roi, ratio, cut_out_mask],...}"""
    def further_image_manipulation(self, r):
        ret = {}
        ret.update({"masks": r["masks"]})
        ret.update({"rois": r["rois"]})
        return ret


class WFN(detection):
    def inference_config(self):
        InferenceConfig = InferenceConfigWFN()
        return InferenceConfig

    def class_names(self):
        return ["bg", "white_window", "white_door", "brick_window", "brick_door"]

    """@return: the Path to the log-directory"""

    def model_dir(self):
        return os.path.join(self.ROOT_DIR, "SMSNetworks/wall_feature_place_detection/" + "logsWallFeatureDetection")

    """@return: the Path to the .h5 file that contains the Head Detection Network"""

    def network_path(self):
        return os.path.join(self.ROOT_DIR, "SMSNetworks/wall_feature_place_detection/" + "mask_rcnn_wall_feature_detection_0100.h5")

    """The classes of WFN are: "bg", "white_window", "white_door", "brick_window", "brick_door". This method returns the id, the masks of the feature places, the rois and the 
            ratio a head covers proportional to the whole picture
            @return: {id: [mask, roi, ratio, cut_out_mask],...}"""

    def further_image_manipulation(self, r):
        return_rois = []
        return_classes = []
        # select door with maximum confidence
        door_confidence = 0
        max_dc_id = 0
        class_names = self.class_names()
        for i, class_id in zip(range(len(r["class_ids"])), r["class_ids"]):
            if class_names[r["class_ids"][i]] == "white_door" or class_names[r["class_ids"][i]] == "brick_door":
                (door_confidence, max_dc_id) = (r["scores"][i], i) if r["scores"][i] > door_confidence else (door_confidence, max_dc_id)
        for i, roi in zip(range(len(r["rois"])), r["rois"]):
            if class_names[r["class_ids"][i]] == "white_door" or class_names[r["class_ids"][i]] == "brick_door":
                if i == max_dc_id:
                    return_rois.append(r["rois"][i])
                    return_classes.append([class_names[r["class_ids"][i]][0: 5],
                                           class_names[r["class_ids"][i]][6: len(class_names[r["class_ids"][i]])]])
            else:
                return_rois.append(r["rois"][i])
                return_classes.append([class_names[r["class_ids"][i]][0: 5],
                                       class_names[r["class_ids"][i]][6: len(class_names[r["class_ids"][i]])]])

        ret = {}
        ret.update({"rois": return_rois})
        ret.update({"class_names": return_classes})
        return ret


def create_face_net_dict_files_for_train_val(picture_path, pic_file):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    pic = skimage.io.imread(picture_path+pic_file)
    ffn = FFN(pic)
    ffn_results = ffn.detect()
    session.close()
    if ffn_results == "redo":
        print("redo")
        return "Failure"
    else:
        output = open("/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/face_net_pkl_val/"+pic_file+".pkl", "wb")
        pkl.dump(ffn_results["face_score_input"], output)
        output.close()
        return "Success"


def scramble_face_net_dicts(face_net_dict_list):
    dict_one_index, dict_two_index = random.randint(0, len(face_net_dict_list)-1), random.randint(0, len(face_net_dict_list)-1)
    print("Chosen_dicts:", dict_one_index, dict_two_index)
    dict_one, dict_two = face_net_dict_list[dict_one_index], face_net_dict_list[dict_two_index]
    face_part_dict = {"inner_eye_l": [["points", "inner_eye_point_array_l"], ["areas", "iris_area_l"], ["points", "iris_center_point_l"], ["areas", "inner_eye_area_l"]],
                      "inner_eye_r": [["points", "inner_eye_point_array_r"], ["areas", "iris_area_r"], ["points", "iris_center_point_r"], ["areas", "inner_eye_area_r"]],
                      "outer_eye_l": [["points", "outer_eye_point_array_l"], ["areas", "outer_eye_area_l"], ["points", "inner_eye_point_array_l"], ["areas", "iris_area_l"], ["points", "iris_center_point_l"], ["areas", "inner_eye_area_l"]],
                      "outer_eye_r": [["points", "outer_eye_point_array_r"], ["areas", "outer_eye_area_r"], ["points", "inner_eye_point_array_r"], ["areas", "iris_area_r"], ["points", "iris_center_point_r"], ["areas", "inner_eye_area_r"]],
                      "nose": [["points", "nose_tip_point_array"]],
                      "mouth": [["points", "mouth_point_array"], ["areas", "mouth_area"]],
                      "cheek_l": [["points", "cheek_point_array_l"]], "cheek_r": [["points", "cheek_point_array_r"]],
                      "chin": [["points", "chin_point_array"]]}
    face_feature_dict_keys = list(face_part_dict.keys())
    scramble_degree = random.randint(1, 6)
    print("Scramble_degree:", scramble_degree)
    for a in range(scramble_degree):
        key = random.choice(face_feature_dict_keys)
        print(key)
        del face_feature_dict_keys[face_feature_dict_keys.index(key)]
        for face_feature in face_part_dict[key]:
            dict_one[face_feature[0]][face_feature[1]] = dict_two[face_feature[0]][face_feature[1]]
    return dict_one


def mirror_face_net_dict(face_net_dict):
    mirror_face_net_dict = {"points": {"iris_center_point_l": None, "iris_center_point_r": None,
                                       "inner_eye_point_array_l": None, "inner_eye_point_array_r": None,
                                       "outer_eye_point_array_l": None, "outer_eye_point_array_r": None,
                                       "cheek_point_array_l": None, "cheek_point_array_r": None,
                                       "eye_brow_point_array_l": None, "eye_brow_point_array_r": None,
                                       "eye_bridge_point": None,
                                       "nose_tip_point_array": None,
                                       "mouth_point_array": None,
                                       "chin_point_array": None},
                            "areas": {"iris_area_l": None, "iris_area_r": None,
                                      "inner_eye_area_l": None, "inner_eye_area_r": None,
                                      "outer_eye_area_l": None, "outer_eye_area_r": None,
                                      "mouth_area": None}
                            }
    eye_bridge_point = face_net_dict["points"]["eye_bridge_point"]
    for point_list_key in face_net_dict["points"]:
        if point_list_key[-1] == "l":
            mirror_point_list_key = point_list_key[0: -1] + "r"
        elif point_list_key[-1] == "r":
            mirror_point_list_key = point_list_key[0: -1] + "l"
        else:
            mirror_point_list_key = point_list_key
        if point_list_key == "iris_center_point_l" or point_list_key == "iris_center_point_r" or point_list_key == "eye_bridge_point":
            point = face_net_dict["points"][point_list_key]
            # eye_bridge_point-(point-eye_bridge_point)
            mirror_face_net_dict["points"][mirror_point_list_key] = [a-b for a, b in zip([point[0], eye_bridge_point[1]], [a-b for a, b in zip([0, point[1]], [0, eye_bridge_point[1]])])]
        else:
            mirrored_point_list = []
            for point in face_net_dict["points"][point_list_key]:
                # eye_bridge_point-(point-eye_bridge_point)
                mirrored_point_list.append([a-b for a, b in zip([point[0], eye_bridge_point[1]], [a-b for a, b in zip([0, point[1]], [0, eye_bridge_point[1]])])])
            mirrored_point_list.reverse()
            mirror_face_net_dict["points"][mirror_point_list_key] = mirrored_point_list
    for area_list_key in face_net_dict["areas"]:
        if area_list_key[-1] == "l":
            mirror_area_list_key = area_list_key[0: -1] + "r"
        elif area_list_key[-1] == "r":
            mirror_area_list_key = area_list_key[0: -1] + "l"
        else:
            mirror_area_list_key = area_list_key
        mirror_face_net_dict["areas"][mirror_area_list_key] = np.flip(face_net_dict["areas"][area_list_key], 1)
    return mirror_face_net_dict


if __name__ == "__main__":
    cuda.select_device(0)
    picture_path = "/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/face_pics_val/"
    pic_files = os.listdir(picture_path)
    failed_pics = []
    for pic_file, i in zip(pic_files, range(len(pic_files))):
        print("Progress:", i+1, "/", len(pic_files), pic_file)
        status = "Success"
        try:
            status = create_face_net_dict_files_for_train_val(picture_path, pic_file)
        except:
            failed_pics.append(picture_path + pic_file)
            print("failed_pics", failed_pics)
        if status == "Failure":
            os.remove(picture_path + pic_file)

    cuda.close()

    """
    changed_file_marker = False
    file_marker = 2
    assert changed_file_marker
    face_net_dicts_file_path = "/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/face_net_jsons/"
    face_net_dict_files = os.listdir(face_net_dicts_file_path)
    face_net_dict_list = []
    face_net_dict = {}
    for face_net_dict_file in face_net_dict_files:
        face_net_score_input_dict = pkl.load(open("/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/face_net_jsons/" + face_net_dict_file, 'rb'))
        face_net_dict_list.append(face_net_score_input_dict)
    scrambled_dicts_amount = 250
    for i in range(scrambled_dicts_amount):
        scrambled_dict = scramble_face_net_dicts(face_net_dict_list)
        file = open("/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/face_net_jsons_scramble/" + str(i)+"-"+str(file_marker)+".pkl", 'wb')
        pkl.dump(scrambled_dict, file)
    """
    """
    face_net_dict_path = "/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/face_net_jsons2/"
    face_net_dict_list = os.listdir(face_net_dict_path)
    for face_net_dict_list_item in face_net_dict_list:
        face_net_dict_file = open(face_net_dict_path + face_net_dict_list_item, 'rb')
        face_net_dict = pkl.load(face_net_dict_file)
        face_net_dict_file.close()
        mirrored_face_net_dict = mirror_face_net_dict(face_net_dict)
        mirror_face_net_dict_file = open(face_net_dict_path + "mirror" + face_net_dict_list_item, 'wb')
        pkl.dump(mirrored_face_net_dict, mirror_face_net_dict_file)
        mirror_face_net_dict_file.close()
    """



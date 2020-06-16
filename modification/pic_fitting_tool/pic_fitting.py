import json
from typing import List, Any, Union

from skimage import io
import os
import sys
import sys
import math
from math import sqrt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def edm_validation(mask, last_direction, current_mid_pixel):
    """runs the most outer circle
        last_directions: right = (0, 1)   guard_pixel: g_0 = (0, 0)
                         left = (0, -1)                g_1 = (0, 2)
                         up = (-1, 0)                  g_2 = (2, 2)
                         down = (1, 0)                 g_3 = (2, 0)

        The next_direction will be correct and it will be one of the directions
        @return: next_direction, new_mid_pixel
        """
    assert mask.shape == (3, 3)
    dir_list = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    dir_val_list = [mask[1][2], mask[1][0], mask[0][1], mask[2][1]]
    guard_pair_val_list = [(mask[0][2], mask[2][2]), (mask[0][0], mask[2][0]),  (mask[0][0], mask[0][2]),
                           (mask[2][0], mask[2][2])]
    curr_max = 0
    curr_max_index = 0
    for i in range(len(dir_list)):
        dir_val_list[i] = dir_val_list[i] * np.subtract(guard_pair_val_list[0], guard_pair_val_list[1])
        if not dir_val_list[i] == 0:
            dir_sum_sqrt = (last_direction[0] + dir_list[i][0]) ^ 2 + (last_direction[1] + dir_list[i][1]) ^ 2
            if dir_sum_sqrt > curr_max:
                curr_max = dir_sum_sqrt
                curr_max_index = i
    next_direction = dir_list[curr_max_index]
    new_mid_pixel = current_mid_pixel + last_direction
    return next_direction, new_mid_pixel


def edge_detection(cut_out_picture, picture_shape, rois):
    start_pixel_array = []
    edge_detection_mask = np.zeros((3, 3))
    edm_shape = edge_detection_mask.shape
    rim_pic = np.zeros((picture_shape[0], picture_shape[1]))
    y_x_array_dict = {-1: [-1]}
    # searching for start pixels
    # rois is an array with bounding-box-pixels: first: y1, x1 for bottom left and scnd: y2, x2 for top right
    for bb in rois:
        for x in range(bb[1], bb[3]):
            start_pixel_found = 0
            for y in range(bb[2], bb[0]):
                if not cut_out_picture[y][x] == 0 | cut_out_picture[y][x] == (0, 0, 0):
                    start_pixel_found = 1
                    start_pixel_array.append((y, x))
                    break
            if start_pixel_found == 1:
                break
    if len(start_pixel_array) == 0:
        return np.zeros(picture_shape), y_x_array_dict

    for start_pixel in start_pixel_array:
        current_mid_pixel = start_pixel
        last_direction = "r"
        while True:
            rim_pic[current_mid_pixel] = 1
            if not y_x_array_dict[current_mid_pixel[0]]:
                y_x_array_dict.update({current_mid_pixel[0]: [[edm_shape[1][0], current_mid_pixel[1], edm_shape[1][2]]]})
            else:
                pixel_array = y_x_array_dict[current_mid_pixel[0]]
                pixel_array.append([edm_shape[1][0], current_mid_pixel[1], edm_shape[1][2]])
                y_x_array_dict.update({current_mid_pixel[0]: pixel_array})
            for y in range(edm_shape[0]):
                for x in range(edm_shape[1]):
                    curr_y, curr_x = current_mid_pixel[0]+y-1, current_mid_pixel[1]+x-1
                    # if curr_y/x falls out of cut_out_picture
                    edge_detection_mask[y][x] = 0 if curr_x < 0 | curr_x >= picture_shape[1] | curr_y < 0 | curr_y >= picture_shape[0] else''
                    edge_detection_mask[y][x] = 1 if not cut_out_picture[curr_y][curr_x] == 0 | cut_out_picture[curr_y][curr_x] == (0, 0, 0) else 0
            last_direction, current_mid_pixel = edm_validation(edge_detection_mask, last_direction, current_mid_pixel)
            # Einen Kreis gelaufen
            if (current_mid_pixel == start_pixel) | (current_mid_pixel - (-1, 0) == start_pixel):
                break
        return rim_pic, y_x_array_dict


class pic_fitting:

    def __init__(self, service):
        assert service in ["over_face"]  # liste aller services
        self.service_path = os.path.join(os.path.dirname(__file__), "services/" + service)
        self.service_pics = [io.imread(pic_name) for pic_name in sorted(listdir(os.path.join(self.service_path, "pic"))) if pic_name[-4: -1] in [".png", ".jpg"]]
        self.pic_mask_dict = json.load(open(os.path.join(self.service_path, "pic_polygon_mask.json")))
        assert len(self.service_pics) == len(self.pic_mask_dict)

    def main_mgm(self, cut_out_picture, picture_shape, rois):
        for k in range(len(rois)):
            mid_pixel = (int((rois[k][0][0] - rois[k][1][0]) / 2), int((rois[k][0][1] - rois[k][1][1]) / 2))
            rim_pic, y_x_array_dict = edge_detection(cut_out_picture, picture_shape, rois[k])
            angle_rimpixel_dict, min_pic_index = self.edge_minimalization(y_x_array_dict, mid_pixel)
            cut_out_picture = self.picture_transformation(cut_out_picture, y_x_array_dict, angle_rimpixel_dict, min_pic_index, mid_pixel)
        return cut_out_picture

    def edge_minimalization(self, y_x_array_dict, mid_pixel):
        zero_degree_vector = mid_pixel - (0, mid_pixel[1])
        len_z_d_v = sqrt(zero_degree_vector[0] ^ 2 + zero_degree_vector[1] ^ 2)
        angle_rimpixel_dict = {}
        divergence_array = []
        for i in range(0, 360):
            angle_rimpixel_dict.update({i: []})
        for ref_pic in range(len(self.pic_mask_dict)):
            ref_angle_rimpixel_dict = self.pic_mask_dict[ref_pic]["angle_rimpixel_dict"]
            divergence_length = 0
            sum_divergence = 0
            for y in y_x_array_dict:
                x_array = y_x_array_dict[y]
                for x in x_array:  # x = [1/0, x_coordinate, 1/0]
                    old_divergence_length = divergence_length
                    rim_pix_vector = np.subtract((y, x[1]), mid_pixel)
                    distance_from_mid = sqrt(rim_pix_vector[0] ^ 2 + rim_pix_vector[1] ^ 2)
                    angle = int(math.acos((rim_pix_vector * zero_degree_vector) / (distance_from_mid * len_z_d_v)))  # 0 <= angle <= 395 element of int
                    divergence_length = min([abs(distance_from_mid - ref_pixel["distance_to_mid"]) for ref_pixel in ref_angle_rimpixel_dict[angle]])  # ref_pixel: {"pixel": (y, x), "distance_to_mid": distance_to_mid_pixel}
                    sum_divergence += abs(divergence_length - old_divergence_length) / divergence_length
            divergence_array.append(sum_divergence)
        min_index = divergence_array.index(min(divergence_array))
        for y in y_x_array_dict:
            x_array = y_x_array_dict[y]
            for x in x_array:
                rim_pix_vector = np.subtract((y, x[1]), mid_pixel)
                distance_from_mid = sqrt(rim_pix_vector[0] ^ 2 + rim_pix_vector[1] ^ 2)
                angle = int(math.acos((rim_pix_vector * zero_degree_vector) / (distance_from_mid * len_z_d_v)))  # 0 <= angle <= 395 element of int
                angle_rimpixel_dict_entry = angle_rimpixel_dict[angle]
                angle_rimpixel_dict_entry.append({"pixel": (y, x[1]), "distance": distance_from_mid})
                angle_rimpixel_dict.update({angle: angle_rimpixel_dict_entry})
        return angle_rimpixel_dict, min_index

    def picture_transformation(self, cut_out_picture, y_x_array_dict, angle_rimpixel_dict, fitting_pic_index, mid_pixel):
        fitting_pic_dict = self.pic_mask_dict[fitting_pic_index]
        service_picture = self.service_pics[fitting_pic_index]
        zero_degree_vector = mid_pixel - (0, mid_pixel[1])
        len_z_d_v = sqrt(zero_degree_vector[0] ^ 2 + zero_degree_vector[1] ^ 2)
        for y in y_x_array_dict:
            x_array = []
            x_array = [x_array.append(np.arange(x[0][1], x[1][1])) if x[0][2] == x[1][0] == 1 else '' for x in zip(y_x_array_dict[y][0: -2], y_x_array_dict[y][1: -1])]
            for x in x_array[0]:
                vector = (y - mid_pixel[0], x - mid_pixel[1])
                vector_len = sqrt(vector[0] ^ 2 + vector[1] ^ 2)
                angle = math.acos((vector * zero_degree_vector) / (vector_len * len_z_d_v))
                min_angle_rimpixel_index, old_angle_dif = 0, 0
                for rim_pixel in angle_rimpixel_dict[int(angle)]:
                    rim_pixel_vector = (mid_pixel[0] - rim_pixel["pixel"][0], mid_pixel[1] - rim_pixel["pixel"][1])
                    real_rim_pixel_angle = math.acos((rim_pixel_vector * zero_degree_vector) / (rim_pixel["distance"] * len_z_d_v))
                    new_angle_dif = abs(angle - real_rim_pixel_angle)
                    min_angle_rimpixel_index += 1 if new_angle_dif < old_angle_dif else''
                    old_angle_dif = new_angle_dif
                chosen_rimpixel = angle_rimpixel_dict[int(angle)][min_angle_rimpixel_index]
                chosen_rimpixel_vector = (mid_pixel[0] - chosen_rimpixel["pixel"][0], mid_pixel[1] - chosen_rimpixel["pixel"][1])
                chosen_rimpixel_vector_length = sqrt(chosen_rimpixel_vector[0] ^ 2 + chosen_rimpixel_vector[1] ^ 2)
                relative_distance = vector_len / chosen_rimpixel_vector_length
                min_angle_rimpixel_index, old_angle_dif = 0, 0
                for service_rim_pixel in fitting_pic_dict["angle_rimpixel_dict"]:
                    service_rim_pixel_vector = (fitting_pic_dict["mid_pixel"][0] - service_rim_pixel["pixel"][0], fitting_pic_dict["mid_pixel"][1] - service_rim_pixel["pixel"][1])
                    real_service_rim_pixel_angle = math.acos((service_rim_pixel_vector * fitting_pic_dict["zero_degree_vector"]) / (service_rim_pixel["distance"] * fitting_pic_dict["len_z_d_v"]))
                    new_angle_dif = abs(angle - real_service_rim_pixel_angle)
                    min_angle_rimpixel_index += 1 if new_angle_dif < old_angle_dif else''
                    old_angle_dif = new_angle_dif
                chosen_service_pixel = fitting_pic_dict["angle_rimpixel_dict"][min_angle_rimpixel_index]
                chosen_service_distance = int(chosen_service_pixel["distance"] * relative_distance)
                relevant_service_pixel_pool = fitting_pic_dict["angle__distance_from_mid_pixel_dict__dict"][chosen_service_distance]
                minimum_service_pixel_index, old_diff_vector_length = 0, 0
                for relevant_pixel in relevant_service_pixel_pool:
                    relevant_vector = np.subtract(fitting_pic_dict["mid_pixel"], relevant_pixel)
                    new_diff_vector = np.subtract(relevant_vector, vector)
                    new_diff_vector_len = sqrt(new_diff_vector[0] ^ 2 + new_diff_vector[1] ^ 2)
                    minimum_service_pixel_index += 1 if new_diff_vector_len < old_diff_vector_length else''
                    old_diff_vector_length = new_diff_vector_len
                most_fitting_service_pixel = relevant_service_pixel_pool[minimum_service_pixel_index]
                cut_out_picture[y][x] = service_picture[most_fitting_service_pixel[0]][most_fitting_service_pixel[1]]
        return cut_out_picture

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
from modification.helper_files import math_helper


def roi_split(roi_img, rows, cols):
    chip_s = [max(math.ceil(roi_img.shape[0] / rows), 1), max(math.ceil(roi_img.shape[1] / cols), 1)]
    img_cp = np.zeros((rows, cols, roi_img[2: -1])) if len(roi_img.shape) > 2 else np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            cp_start = [r * chip_s[0], c * chip_s[1]]
            for ir in range(chip_s[0]):
                for jc in range(chip_s[1]):
                    if (cp_start[0] + ir) < roi_img.shape[0] and (cp_start[1] + jc) < roi_img.shape[1]:
                        img_cp[r, c] = img_cp[r, c] + roi_img[cp_start[0] + ir, cp_start[1] + jc]
    # plt.imshow(img_cp)
    # plt.show()
    return img_cp


def find_center(roi_image):
    left, right, up, down = find_outermost_line(roi_image, "left"), find_outermost_line(roi_image, "right")+1, find_outermost_line(roi_image, "up"), find_outermost_line(roi_image, "down")+1
    return [(up + down) // 2, (left + right) // 2]


def find_cog(roi_image):
    y_gravity, x_gravity = find_side_centric_point(roi_image, "left", gravity=1)[0], find_side_centric_point(roi_image, "up", gravity=1)[1]
    #print(y_gravity, x_gravity)
    return [y_gravity, x_gravity]


def find_outermost_line(roi_image, side):
    side_array = list(filter(lambda x: not (x[0] == x[1] == 0), [find_most_outer_points_of_array(roi_image[i]) for i in range(roi_image.shape[0])])) if side == "left" or side == "right" else \
                 list(filter(lambda x: not (x[0] == x[1] == 0), [find_most_outer_points_of_array(roi_image[:, i]) for i in range(roi_image.shape[1])]))
    if side_array:
        if side == "left":
            return min([side_array_element[0] for side_array_element in side_array])
        elif side == "right":
            return max([side_array_element[1] for side_array_element in side_array])
        elif side == "up":
            return min([side_array_element[0] for side_array_element in side_array])
        elif side == "down":
            return max([side_array_element[1] for side_array_element in side_array])
    else:
        return 0


def find_side_centric_point(roi_image, side, gravity=None):
    gravity = 0.1 if gravity is None else gravity
    assert 0 <= gravity <= 1
    opposite_side = {"left": "right", "right": "left", "up": "down", "down": "up"}
    # print(roi_image.shape, side, "----------------------")
    outermost_line, opp_outermost_line = find_outermost_line(roi_image, side), find_outermost_line(roi_image, opposite_side[side])
    # print(outermost_line, opp_outermost_line, (outermost_line - opp_outermost_line) * gravity, "gravity", gravity)
    mid_point_array = []
    _range = [outermost_line, outermost_line + int(abs(outermost_line - opp_outermost_line) * gravity)+1] if side == "left" or side == "up" else [outermost_line - int(abs(outermost_line - opp_outermost_line) * gravity), outermost_line+1]
    #print("range", _range)
    for i in range(_range[0], _range[1]):
        roi_image_line = roi_image[:, i] if side == "left" or side == "right" else roi_image[i]
        point_lu, point_rd = find_most_outer_points_of_array(roi_image_line)
        # print(point_lu, point_rd)
        mid_point_array.append((point_lu + point_rd) // 2)
    #print(mid_point_array)
    resulting_mid_point = np.sum(mid_point_array) // len(mid_point_array)
    # print([resulting_mid_point, outermost_line]) if side == "left" or side == "right" else print([outermost_line, resulting_mid_point, side])
    return [resulting_mid_point, outermost_line] if side == "left" or side == "right" else [outermost_line, resulting_mid_point]


def cut_out_roi(image, roi):
    yx_cutted_image = np.zeros((roi[2] - roi[0], roi[3] - roi[1], 3), dtype=np.uint8) if len(image.shape) > 2 else np.zeros((roi[2] - roi[0], roi[3] - roi[1]), dtype=np.uint8)
    y_cutted_image = image[roi[0]: roi[2]]
    if len(image.shape) > 2:
        for i in range(roi[2] - roi[0]):
            yx_cutted_image[i, :, :] = y_cutted_image[i][roi[1]: roi[3]]
    else:
        for i in range(roi[2] - roi[0]):
            yx_cutted_image[i] = y_cutted_image[i][roi[1]: roi[3]]
    #yx_cutted_image = image[roi[0]: roi[2], roi[1]: roi[3]]
    #plt.imshow(yx_cutted_image)
    #plt.title(yx_cutted_image.shape)
    #plt.show()
    return yx_cutted_image


def find_most_outer_points_of_array(array):

    point_lu, point_rd = 0, 0
    plu_found = False
    if np.sum(array[0]) > 0 and np.sum(array[-1] > 0):
        return [0, len(array)-1]
    for j in range(0, len(array)-1):
        #print(array[j], array[j-1],"-----------------------")
        if not plu_found and np.subtract(array[j], array[j-1]) == array[j] and array[j] > 0:
            point_lu = j
            plu_found = True
        if np.subtract(array[j], array[j + 1]) == array[j] and array[j] > 0:
            point_rd = j
    if np.sum(array[0]) > 0:
        point_lu = 0
    if np.sum(array[-1]) > 0:
        point_rd = len(array)-1
    return [point_lu, point_rd]



def find_points_on_gravity_line(roi_image, line_layout, point_amount):
    assert line_layout in [["left", "right"], ["up", "down"]]
    outermost_line_lu, outermost_line_rd = find_outermost_line(roi_image, line_layout[0]), find_outermost_line(roi_image, line_layout[1])
    chunk_length = abs(outermost_line_rd - outermost_line_lu) / point_amount
    # print(chunk_length)
    points_on_gravity_line = []
    for i in range(0, point_amount):
        roi_img_chunk = roi_image[:, outermost_line_lu + int(i*chunk_length): outermost_line_lu + int((i+1)*chunk_length)] if line_layout == ["left", "right"] else roi_image[outermost_line_lu + int(i*chunk_length): outermost_line_lu + int((i+1)*chunk_length)]
        #print(roi_img_chunk.shape)
        #cog_of_chunk = find_cog(roi_image[:, outermost_line_lu + int(i*chunk_length): outermost_line_lu + int((i+1)*chunk_length)]) if line_layout == ["left", "right"] else find_cog(roi_image[outermost_line_lu + int(i*chunk_length): outermost_line_lu + int((i+1)*chunk_length)])
        cog_of_chunk = [a+b for a, b in zip(find_cog(roi_img_chunk), [0, int(i*chunk_length)])]
        points_on_gravity_line.append(cog_of_chunk)
    #print(points_on_gravity_line)
    return points_on_gravity_line


def find_area_diff(roi_image1, roi_image2):
    return np.subtract(roi_image1, roi_image2)


def find_area_diff_bool(roi_image1, roi_image2):
    diff_image = find_area_diff(roi_image1, roi_image2)
    return [[1 if not (diff_image[y][x] == 0 or diff_image[y][x] == (0, 0) or diff_image[y][x] == (0, 0, 0)) else 0 for x in diff_image.shape[1]] for y in diff_image.shape[0]]


def area_sum(roi_image):
    return sum([sum(line) for line in roi_image])

from json import dumps, load
import pprint
from typing import List, Any, Union
from skimage import io
import os
import sys
import math
from math import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import codecs
from modification.helper_files import roi_helper


def calculate_rotation(three_point_array):
    """Errechnet die Rotation anhand der relativen Länge zwischen (p0, p1) und (p1, p2).
       Es geht davon aus, dass bei 0° Rotation (p0, p1) == (p1, p2) und dass die Höhe von p1 über (p0, p2) genauso groß ist, wie (p0, p2)/2"""
    return degrees(atan(np.subtract(three_point_array[1][1], three_point_array[0][1])/np.subtract(three_point_array[2][1], three_point_array[1][1])) - pi/4)


def calculate_angle_rad(p1, p2, p3):
    """p1 ist in der Mitte vom Winkel"""
    v1 = calculate_vector(p1, p2)
    v2 = calculate_vector(p1, p3)
    return angle_rad_between_vector(v1, v2)


def calculate_angle_deg(p1, p2, p3):
    """p1 ist in der Mitte vom Winkel"""
    v1 = calculate_vector(p1, p2)
    v2 = calculate_vector(p1, p3)
    return angle_deg_between_vector(v1, v2)


def calculate_line(p1, p2):
    """vector geht von p1 nach p2"""
    vector = calculate_vector(p2, p1)
    length = vector_length(vector)
    return [vector, length]


def dot_product(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))


def vector_length(v):
    return sqrt(dot_product(v, v))


def calculate_vector(p_start, p_end):
    """vector geht von p_start nach p_end, der return_vector fängt bei [0, 0] an"""
    return np.array([a-b for a, b in zip(p_end, p_start)])


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    if vector_length(vector) == 0:
        return 0
    else:
        return vector / vector_length(vector)


def angle_rad_between_vector(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_deg_between_vector(v1, v2):
    return degrees(angle_rad_between_vector(v1, v2))


def calculate_area_rot(area, f_h, degree):
    """rotates only around the center x_axis, so it rotates only in the x, z plane"""
    img = np.zeros((area.shape[0], int(area.shape[1]*3)))
    shape_diff = [img.shape[0] - area.shape[0], img.shape[1] - area.shape[1]]
    #print(shape_diff)
    area_anchor = [int(shape_diff[0]//2), int(shape_diff[1]//2)]
    #rot_c = [int(area.shape[1]//2), 0]  # rot_c has axis x, z but NOT y
    rot_c = [0, 0]
    for y in range(area.shape[0]):
        for x in range(area.shape[1]):
            xz_v = [x, f_h(x)]
            #print(rot_c)
            rc_xz_v = calculate_vector(rot_c, xz_v)
            rot_p = [y, int(calculate_point_rot(rc_xz_v, degree)[0])]
            img[area_anchor[0]+rot_p[0], area_anchor[1]+rot_p[1]] = area[y, x]
    for y in range(1, img.shape[0]-1):
        for x in range(1, img.shape[1]-1):
            if img[y, x-1] > img[y, x] < img[y, x+1]:
                img[y, x] = img[y, x+1]
    cut_img = roi_helper.cut_out_roi(img, [roi_helper.find_outermost_line(img, "up"), roi_helper.find_outermost_line(img, "left"), roi_helper.find_outermost_line(img, "down"), roi_helper.find_outermost_line(img, "right")])

    return cut_img


def calculate_point_rot(vector, degree):
    """rotates only in the x, z plane"""
    # print(vector, "vector")
    rot_m = np.asarray([[cos(radians(degree)), -sin(radians(degree))],
                        [sin(radians(degree)), cos(radians(degree))]])
    # print(rot_m.dot(vector))
    return rot_m.dot(vector)

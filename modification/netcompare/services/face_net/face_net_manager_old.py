from json import dumps, load
import pprint
from typing import List, Any, Union
from skimage import io
from skimage.draw import line
import os
import sys
import math
from math import sqrt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import codecs

BLACK_PICTURE_0_DEGREE = np.zeros((540, 380))

raw_points_0_degree = {
    "iris_center_point_l": [[265, 100]],
    "iris_center_point_r": [[265, 280]],
    "cheek_point_array_l": [[310, 44], [340, 94]],
    "cheek_point_array_r": [[340, 286], [310, 335]],
    "inner_eye_point_array_l": [[265, 64], [265, 136]],
    "inner_eye_point_array_r": [[265, 244], [265, 316]],
    "outer_eye_point_array_l": [[265, 48], [265, 153]],
    "outer_eye_point_array_r": [[265, 227], [265, 332]],
    "eye_brow_point_array_l": [[235, 47], [220, 74], [214, 100], [220, 121], [235, 143]],
    "eye_brow_point_array_r": [[235, 237], [220, 259], [214, 280], [220, 306], [235, 332]],
    "eye_bridge_point": [[265, 190]],
    "nose_point_array": [[370, 136], [355, 190], [370, 244]],
    "mouth_point_array": [[445, 100], [445, 190], [445, 280]],
    "chin_point_array": [[535, 154], [505, 190], [535, 226]],
}

raw_face_net_0_degree = {
    "point_heights": {
        "iris_center_point_l": [6],
        "iris_center_point_r": [6],
        "cheek_point_array_l": [-75, 15],
        "cheek_point_array_r": [15, -75],
        "inner_eye_point_array_l": [-45, 0],
        "inner_eye_point_array_r": [0, -45],
        "outer_eye_point_array_l": [-54, 6],
        "outer_eye_point_array_r": [6, -54],
        "eye_brow_point_array_l": [-54, 6, 36, 66, 51],
        "eye_brow_point_array_r": [51, 66, 36, 6, -54],
        "eye_bridge_point": [0],
        "nose_point_array": [30, 120, 30],
        "mouth_point_array": [0, 90, 0],
        "chin_point_array": [15, 36, 15],
    },
    "points": raw_points_0_degree,
    "lines": {
        "ebpal0_ebpal1": [raw_points_0_degree["eye_brow_point_array_l"][0], raw_points_0_degree["eye_brow_point_array_l"][1]],
        "ebpal1_ebpal2": [raw_points_0_degree["eye_brow_point_array_l"][1], raw_points_0_degree["eye_brow_point_array_l"][2]],
        "ebpal2_ebpal3": [raw_points_0_degree["eye_brow_point_array_l"][2], raw_points_0_degree["eye_brow_point_array_l"][3]],
        "ebpal3_ebpal4": [raw_points_0_degree["eye_brow_point_array_l"][3], raw_points_0_degree["eye_brow_point_array_l"][4]],
        "ebpar0_ebpar1": [raw_points_0_degree["eye_brow_point_array_r"][0], raw_points_0_degree["eye_brow_point_array_r"][1]],
        "ebpar1_ebpar2": [raw_points_0_degree["eye_brow_point_array_r"][1], raw_points_0_degree["eye_brow_point_array_r"][2]],
        "ebpar2_ebpar3": [raw_points_0_degree["eye_brow_point_array_r"][2], raw_points_0_degree["eye_brow_point_array_r"][3]],
        "ebpar3_ebpar4": [raw_points_0_degree["eye_brow_point_array_r"][3], raw_points_0_degree["eye_brow_point_array_r"][4]],
        "oepal0_oepal1": [raw_points_0_degree["outer_eye_point_array_l"][0], raw_points_0_degree["outer_eye_point_array_l"][1]],
        "oepar0_oepar1": [raw_points_0_degree["outer_eye_point_array_r"][0], raw_points_0_degree["outer_eye_point_array_r"][1]],
        "iepal0_iepal1": [raw_points_0_degree["inner_eye_point_array_l"][0], raw_points_0_degree["inner_eye_point_array_l"][1]],
        "iepar0_iepar1": [raw_points_0_degree["inner_eye_point_array_r"][0], raw_points_0_degree["inner_eye_point_array_r"][1]],
        "oepal0_iepal0": [raw_points_0_degree["outer_eye_point_array_l"][0], raw_points_0_degree["inner_eye_point_array_l"][0]],
        "iepar1_oepar1": [raw_points_0_degree["inner_eye_point_array_r"][1], raw_points_0_degree["outer_eye_point_array_r"][1]],
        "iepal1_oepal1": [raw_points_0_degree["inner_eye_point_array_l"][1], raw_points_0_degree["outer_eye_point_array_l"][1]],
        "oepar0_iepar0": [raw_points_0_degree["outer_eye_point_array_r"][0], raw_points_0_degree["inner_eye_point_array_r"][0]],
        "iepal0_icpl": [raw_points_0_degree["inner_eye_point_array_l"][0], raw_points_0_degree["iris_center_point_l"][0]],
        "icpl_iepal1": [raw_points_0_degree["iris_center_point_l"][0], raw_points_0_degree["inner_eye_point_array_l"][1]],
        "iepar0_icpr": [raw_points_0_degree["inner_eye_point_array_r"][0], raw_points_0_degree["iris_center_point_r"][0]],
        "icpr_iepar1": [raw_points_0_degree["iris_center_point_r"][0], raw_points_0_degree["inner_eye_point_array_r"][1]],
        "oepal1_ebp": [raw_points_0_degree["outer_eye_point_array_l"][1], raw_points_0_degree["eye_bridge_point"][0]],
        "ebp_oepar0": [raw_points_0_degree["eye_bridge_point"][0], raw_points_0_degree["outer_eye_point_array_r"][0]],
        "ebpal4_ebp": [raw_points_0_degree["eye_brow_point_array_l"][4], raw_points_0_degree["eye_bridge_point"][0]],
        "ebp_ebpar0": [raw_points_0_degree["eye_bridge_point"][0], raw_points_0_degree["eye_brow_point_array_r"][0]],
        "oepal0_cepal0": [raw_points_0_degree["outer_eye_point_array_l"][0], raw_points_0_degree["cheek_point_array_l"][0]],
        "oepar1_cepar1": [raw_points_0_degree["outer_eye_point_array_r"][1], raw_points_0_degree["cheek_point_array_r"][1]],
        "oepal0_cepal1": [raw_points_0_degree["outer_eye_point_array_l"][0], raw_points_0_degree["cheek_point_array_l"][1]],
        "oepar1_cepar0": [raw_points_0_degree["outer_eye_point_array_r"][1], raw_points_0_degree["cheek_point_array_r"][0]],
        "cepal0_cepal1": [raw_points_0_degree["cheek_point_array_l"][0], raw_points_0_degree["cheek_point_array_l"][1]],
        "cepar0_cepar1": [raw_points_0_degree["cheek_point_array_r"][0], raw_points_0_degree["cheek_point_array_r"][1]],
        "cepal1_oepal1": [raw_points_0_degree["cheek_point_array_l"][1], raw_points_0_degree["outer_eye_point_array_l"][1]],
        "oepar0_cepar0": [raw_points_0_degree["outer_eye_point_array_r"][0], raw_points_0_degree["cheek_point_array_r"][0]],
        "cepal1_npa0": [raw_points_0_degree["cheek_point_array_l"][1], raw_points_0_degree["nose_point_array"][0]],
        "npa2_cepar0": [raw_points_0_degree["nose_point_array"][2], raw_points_0_degree["cheek_point_array_r"][0]],
        "npa0_oepal1": [raw_points_0_degree["nose_point_array"][0], raw_points_0_degree["outer_eye_point_array_l"][1]],
        "oepar0_npa2": [raw_points_0_degree["outer_eye_point_array_r"][0], raw_points_0_degree["nose_point_array"][2]],
        "npa0_ebp": [raw_points_0_degree["nose_point_array"][0], raw_points_0_degree["eye_bridge_point"][0]],
        "ebp_npa2": [raw_points_0_degree["eye_bridge_point"][0], raw_points_0_degree["nose_point_array"][2]],
        "npa0_npa2": [raw_points_0_degree["nose_point_array"][0], raw_points_0_degree["nose_point_array"][2]],
        "ebp_npa1": [raw_points_0_degree["eye_bridge_point"][0], raw_points_0_degree["nose_point_array"][1]],
        "npa1_mpa1": [raw_points_0_degree["nose_point_array"][1], raw_points_0_degree["mouth_point_array"][1]],
        "mpa0_npa0": [raw_points_0_degree["mouth_point_array"][0], raw_points_0_degree["nose_point_array"][0]],
        "npa2_mpa2": [raw_points_0_degree["nose_point_array"][2], raw_points_0_degree["mouth_point_array"][2]],
        "mpa0_mpa1": [raw_points_0_degree["mouth_point_array"][0], raw_points_0_degree["mouth_point_array"][1]],
        "mpa1_mpa2": [raw_points_0_degree["mouth_point_array"][1], raw_points_0_degree["mouth_point_array"][2]],
        "npa0_cipa0": [raw_points_0_degree["nose_point_array"][0], raw_points_0_degree["chin_point_array"][0]],
        "npa0_npa1": [raw_points_0_degree["nose_point_array"][0], raw_points_0_degree["nose_point_array"][1]],
        "npa1_npa2": [raw_points_0_degree["nose_point_array"][1], raw_points_0_degree["nose_point_array"][2]],
        "cipa2_npa2": [raw_points_0_degree["chin_point_array"][2], raw_points_0_degree["nose_point_array"][2]],
        "mpa1_cipa1": [raw_points_0_degree["mouth_point_array"][1], raw_points_0_degree["chin_point_array"][1]],
        "cipa0_cipa1": [raw_points_0_degree["chin_point_array"][0], raw_points_0_degree["chin_point_array"][1]],
        "cipa1_cipa2": [raw_points_0_degree["chin_point_array"][1], raw_points_0_degree["chin_point_array"][2]],
        "cipa0_cipa2": [raw_points_0_degree["chin_point_array"][0], raw_points_0_degree["chin_point_array"][2]],
        "cepal1_mpa0": [raw_points_0_degree["cheek_point_array_l"][1], raw_points_0_degree["mouth_point_array"][0]],
        "mpa2_cepar0": [raw_points_0_degree["mouth_point_array"][2], raw_points_0_degree["cheek_point_array_r"][0]],
        "oepal0_mpa0": [raw_points_0_degree["outer_eye_point_array_l"][0], raw_points_0_degree["mouth_point_array"][0]],
        "mpa2_oepar1": [raw_points_0_degree["mouth_point_array"][2], raw_points_0_degree["outer_eye_point_array_r"][1]],
        "mpa0_cipa0": [raw_points_0_degree["mouth_point_array"][0], raw_points_0_degree["chin_point_array"][0]],
        "cipa2_mpa2": [raw_points_0_degree["chin_point_array"][2], raw_points_0_degree["mouth_point_array"][2]],
    },
    "areas": {
        "iris_area_l": [],
        "iris_area_r": [],
        "inner_eye_area_l": [],
        "inner_eye_area_r": [],
        "outer_eye_area_l": [],
        "outer_eye_area_r": [],
        "mouth_area": [],
    },
    "angles": {
        "oepal0_iepal0_icpl": [raw_points_0_degree["outer_eye_point_array_l"][0],
                               raw_points_0_degree["inner_eye_point_array_l"][0],
                               raw_points_0_degree["iris_center_point_l"][0]],
        "iepal0_icpl_iepal1": [raw_points_0_degree["inner_eye_point_array_l"][0], raw_points_0_degree["iris_center_point_l"][0],
                               raw_points_0_degree["inner_eye_point_array_l"][1]],
        "icpl_iepal1_oepal1": [raw_points_0_degree["iris_center_point_l"][0], raw_points_0_degree["inner_eye_point_array_l"][1],
                               raw_points_0_degree["outer_eye_point_array_l"][1]],
        "iepal1_oepal1_ebp": [raw_points_0_degree["inner_eye_point_array_l"][1],
                              raw_points_0_degree["outer_eye_point_array_l"][1], raw_points_0_degree["eye_bridge_point"][0]],
        "oepal1_ebp_npa1": [raw_points_0_degree["outer_eye_point_array_l"][1], raw_points_0_degree["eye_bridge_point"][0],
                            raw_points_0_degree["nose_point_array"][1]],
        "npa1_ebp_oepar0": [raw_points_0_degree["nose_point_array"][1], raw_points_0_degree["eye_bridge_point"][0],
                            raw_points_0_degree["outer_eye_point_array_r"][0]],
        "ebp_oepar0_iepar0": [raw_points_0_degree["eye_bridge_point"][0], raw_points_0_degree["outer_eye_point_array_r"][0],
                              raw_points_0_degree["inner_eye_point_array_r"][0]],
        "oepar0_iepar0_icpr": [raw_points_0_degree["outer_eye_point_array_r"][0],
                               raw_points_0_degree["inner_eye_point_array_r"][0],
                               raw_points_0_degree["iris_center_point_r"][0]],
        "iepar0_icpr_iepar1": [raw_points_0_degree["inner_eye_point_array_r"][0], raw_points_0_degree["iris_center_point_r"][0],
                               raw_points_0_degree["inner_eye_point_array_r"][1]],
        "icpr_iepar1_oepar1": [raw_points_0_degree["iris_center_point_r"][0], raw_points_0_degree["inner_eye_point_array_r"][1],
                               raw_points_0_degree["outer_eye_point_array_r"][1]],
        "ebpal4_ebp_oepal1": [raw_points_0_degree["eye_brow_point_array_l"][4], raw_points_0_degree["eye_bridge_point"][0],
                              raw_points_0_degree["outer_eye_point_array_l"][1]],
        "ebpar0_ebp_oepar0": [raw_points_0_degree["eye_brow_point_array_l"][0], raw_points_0_degree["eye_bridge_point"][0],
                              raw_points_0_degree["outer_eye_point_array_r"][0]],
        "oepal0_cepal0_cepal1": [raw_points_0_degree["outer_eye_point_array_l"][0],
                                 raw_points_0_degree["cheek_point_array_l"][0], raw_points_0_degree["cheek_point_array_l"][1]],
        "oepal0_cepal1_oepal1": [raw_points_0_degree["outer_eye_point_array_l"][0],
                                 raw_points_0_degree["cheek_point_array_l"][1],
                                 raw_points_0_degree["outer_eye_point_array_l"][1]],
        "oepal1_npa0_npa2": [raw_points_0_degree["outer_eye_point_array_l"][1], raw_points_0_degree["nose_point_array"][0],
                             raw_points_0_degree["nose_point_array"][2]],
        "npa0_ebp_npa1": [raw_points_0_degree["nose_point_array"][0], raw_points_0_degree["eye_bridge_point"][0],
                          raw_points_0_degree["nose_point_array"][1]],
        "npa1_ebp_npa2": [raw_points_0_degree["nose_point_array"][1], raw_points_0_degree["eye_bridge_point"][0],
                          raw_points_0_degree["nose_point_array"][2]],
        "npa0_npa1_mpa1": [raw_points_0_degree["nose_point_array"][0], raw_points_0_degree["nose_point_array"][1],
                           raw_points_0_degree["mouth_point_array"][1]],
        "mpa1_npa1_npa2": [raw_points_0_degree["mouth_point_array"][1], raw_points_0_degree["nose_point_array"][1],
                           raw_points_0_degree["nose_point_array"][2]],
        "npa0_npa2_oepar0": [raw_points_0_degree["nose_point_array"][0], raw_points_0_degree["nose_point_array"][2],
                             raw_points_0_degree["outer_eye_point_array_r"][0]],
        "oepar0_npa2_cepar0": [raw_points_0_degree["outer_eye_point_array_r"][0], raw_points_0_degree["nose_point_array"][2],
                               raw_points_0_degree["cheek_point_array_r"][0]],
        "cepal1_npa0_oepal1": [raw_points_0_degree["cheek_point_array_l"][1], raw_points_0_degree["nose_point_array"][0],
                               raw_points_0_degree["outer_eye_point_array_l"][1]],
        "oepar0_cepar0_oepar1": [raw_points_0_degree["outer_eye_point_array_r"][0],
                                 raw_points_0_degree["cheek_point_array_r"][0],
                                 raw_points_0_degree["outer_eye_point_array_r"][1]],
        "cepar0_cepar1_oepar1": [raw_points_0_degree["cheek_point_array_r"][0], raw_points_0_degree["cheek_point_array_r"][1],
                                 raw_points_0_degree["outer_eye_point_array_r"][1]],
        "oepal0_mpa0_mpa1": [raw_points_0_degree["outer_eye_point_array_l"][0], raw_points_0_degree["mouth_point_array"][0],
                             raw_points_0_degree["mouth_point_array"][1]],
        "mpa1_mpa2_oepar1": [raw_points_0_degree["mouth_point_array"][1], raw_points_0_degree["mouth_point_array"][2],
                             raw_points_0_degree["outer_eye_point_array_r"][1]],
        "npa0_mpa0_mpa1": [raw_points_0_degree["nose_point_array"][0], raw_points_0_degree["mouth_point_array"][0],
                           raw_points_0_degree["mouth_point_array"][1]],
        "mpa1_mpa2_npa2": [raw_points_0_degree["mouth_point_array"][1], raw_points_0_degree["mouth_point_array"][2],
                           raw_points_0_degree["nose_point_array"][2]],
        "mpa0_mpa1_npa1": [raw_points_0_degree["mouth_point_array"][0], raw_points_0_degree["mouth_point_array"][1],
                           raw_points_0_degree["nose_point_array"][1]],
        "npa1_mpa1_mpa2": [raw_points_0_degree["nose_point_array"][1], raw_points_0_degree["mouth_point_array"][1],
                           raw_points_0_degree["mouth_point_array"][2]],
        "cipa0_mpa0_mpa1": [raw_points_0_degree["chin_point_array"][0], raw_points_0_degree["mouth_point_array"][0],
                            raw_points_0_degree["mouth_point_array"][1]],
        "mpa1_mpa2_cipa2": [raw_points_0_degree["mouth_point_array"][1], raw_points_0_degree["mouth_point_array"][2],
                            raw_points_0_degree["chin_point_array"][2]],
        "ebp_npa1_mpa1": [raw_points_0_degree["eye_bridge_point"][0], raw_points_0_degree["nose_point_array"][1],
                          raw_points_0_degree["mouth_point_array"][1]],
        "npa1_mpa1_cipa1": [raw_points_0_degree["nose_point_array"][1], raw_points_0_degree["mouth_point_array"][1],
                            raw_points_0_degree["chin_point_array"][1]],
    },
    "line_compare": {
        "oepal0_oepal1@ref": ["oepal0_oepal1", "@ref"],
        "oepar0_oepar1@ref": ["oepar0_oepar1", "@ref"],
        "iepal0_iepal1@ref": ["iepal0_iepal1", "@ref"],
        "iepar0_iepar1@ref": ["iepar0_iepar1", "@ref"],
        "oepal0_iepal0@ref": ["oepal0_iepal0", "@ref"],
        "iepar1_oepar1@ref": ["iepar1_oepar1", "@ref"],
        "iepal1_oepal1@ref": ["iepal1_oepal1", "@ref"],
        "oepar0_iepar0@ref": ["oepar0_iepar0", "@ref"],
        "iepal0_icpl@ref": ["iepal0_icpl", "@ref"],
        "icpl_iepal1@ref": ["icpl_iepal1", "@ref"],
        "iepar0_icpr@ref": ["iepar0_icpr", "@ref"],
        "icpr_iepar1@ref": ["icpr_iepar1", "@ref"],
        "oepal1_ebp@ref": ["oepal1_ebp", "@ref"],
        "ebp_oepar0@ref": ["ebp_oepar0", "@ref"],
        "ebpal4_ebp@ref": ["ebpal4_ebp", "@ref"],
        "ebp_ebpar0@ref": ["ebp_ebpar0", "@ref"],
        "oepal0_cepal0@ref": ["oepal0_cepal0", "@ref"],
        "oepar1_cepar1@ref": ["oepar1_cepar1", "@ref"],
        "oepal0_cepal1@ref": ["oepal0_cepal1", "@ref"],
        "oepar1_cepar0@ref": ["oepar1_cepar0", "@ref"],
        "cepal0_cepal1@ref": ["cepal0_cepal1", "@ref"],
        "cepar0_cepar1@ref": ["cepar0_cepar1", "@ref"],
        "cepal1_oepal1@ref": ["cepal1_oepal1", "@ref"],
        "oepar0_cepar0@ref": ["oepar0_cepar0", "@ref"],
        "cepal1_npa0@ref": ["cepal1_npa0", "@ref"],
        "npa2_cepar0@ref": ["npa2_cepar0", "@ref"],
        "npa0_oepal1@ref": ["npa0_oepal1", "@ref"],
        "oepar0_npa2@ref": ["oepar0_npa2", "@ref"],
        "npa0_ebp@ref": ["npa0_ebp", "@ref"],
        "ebp_npa2@ref": ["ebp_npa2", "@ref"],
        "npa0_npa2@ref": ["npa0_npa2", "@ref"],
        "ebp_npa1@ref": ["ebp_npa1", "@ref"],
        "npa1_mpa1@ref": ["npa1_mpa1", "@ref"],
        "mpa0_npa0@ref": ["mpa0_npa0", "@ref"],
        "npa2_mpa2@ref": ["npa2_mpa2", "@ref"],
        "mpa0_mpa1@ref": ["mpa0_mpa1", "@ref"],
        "mpa1_mpa2@ref": ["mpa1_mpa2", "@ref"],
        "npa0_cipa0@ref": ["npa0_cipa0", "@ref"],
        "cipa2_npa2@ref": ["cipa2_npa2", "@ref"],
        "mpa1_cipa1@ref": ["mpa1_cipa1", "@ref"],
        "cipa0_cipa1@ref": ["cipa0_cipa1", "@ref"],
        "cipa1_cipa2@ref": ["cipa1_cipa2", "@ref"],
        "cipa0_cipa2@ref": ["cipa0_cipa2", "@ref"],
        "cepal1_mpa0@ref": ["cepal1_mpa0", "@ref"],
        "mpa2_cepar0@ref": ["mpa2_cepar0", "@ref"],
        "oepal0_mpa0@ref": ["oepal0_mpa0", "@ref"],
        "mpa2_oepar1@ref": ["mpa2_oepar1", "@ref"],
        "oepal0_oepal1_oepar0_oepar1": ["oepal0_oepal1", "oepar0_oepar1"],
        "iepal0_iepal1_iepar0_iepar1": ["iepal0_iepal1", "iepar0_iepar1"],
        "oepal0_iepal0_iepar1_oepar1": ["oepal0_iepal0", "iepar1_oepar1"],
        "iepal1_oepal1_oepar0_iepar0": ["iepal1_oepal1", "oepar0_iepar0"],
        "iepal0_icpl_iepar0_icpr": ["iepal0_icpl", "iepar0_icpr"],
        "icpl_iepal1_icpr_iepar1": ["icpl_iepal1", "icpr_iepar1"],
        "oepal1_ebp_ebp_oepar0": ["oepal1_ebp", "ebp_oepar0"],
        "ebpal4_ebp_ebp_ebpar0": ["ebpal4_ebp", "ebp_ebpar0"],
        "oepal0_cepal0_oepar1_cepar1": ["oepal0_cepal0", "oepar1_cepar1"],
        "oepal0_cepal1_oepar1_cepar0": ["oepal0_cepal1", "oepar1_cepar0"],
        "cepal1_oepal1_oepar0_cepar0": ["cepal1_oepal1", "oepar0_cepar0"],
        "cepal1_npa0_npa2_cepar0": ["cepal1_npa0", "npa2_cepar0"],
        "npa0_oepal1_oepar0_npa2": ["npa0_oepal1", "oepar0_npa2"],
        "npa0_ebp_ebp_npa2": ["npa0_ebp", "ebp_npa2"],
        "ebp_npa1_npa1_mpa1": ["ebp_npa1", "npa1_mpa1"],
        "npa0_npa1_npa1_npa2": ["npa0_npa1", "npa1_npa2"],
        "mpa0_npa0_npa2_mpa2": ["mpa0_npa0", "npa2_mpa2"],
        "mpa0_mpa1_mpa1_mpa2": ["mpa0_mpa1", "mpa1_mpa2"],
        "npa0_cipa0_cipa2_npa2": ["npa0_cipa0", "cipa2_npa2"],
        "mpa0_cipa0_cipa2_mpa2": ["mpa0_cipa0", "cipa2_mpa2"],
        "npa1_mpa1_mpa1_cipa1": ["npa1_mpa1", "mpa1_cipa1"],
        "cipa0_cipa1_cipa1_cipa2": ["cipa0_cipa1", "cipa1_cipa2"],
        "cepal1_mpa0_mpa2_cepar0": ["cepal1_mpa0", "mpa2_cepar0"],
        "oepal0_mpa0_mpa2_oepar1": ["oepal0_mpa0", "mpa2_oepar1"],
    },
    "area_compare": {
        "iris_area_l_@ref": ["iris_area_l", "@ref"],
        "iris_area_r_@ref": ["iris_area_r", "@ref"],
        "inner_eye_area_l_@ref": ["inner_eye_area_l", "@ref"],
        "inner_eye_area_r_@ref": ["inner_eye_area_r", "@ref"],
        "outer_eye_area_l_@ref": ["outer_eye_area_l", "@ref"],
        "outer_eye_area_r_@ref": ["outer_eye_area_r", "@ref"],
        "mouth_area_@ref": ["mouth_area", "@ref"],
        "iris_area_l_iris_are_r": ["iris_area_l", "iris_are_r"],
        "inner_eye_area_l_inner_eye_area_r": ["inner_eye_area_l", "inner_eye_area_r"],
        "outer_eye_area_l_outer_eye_area_r": ["outer_eye_area_l", "outer_eye_area_r"],
        "mouth_area_l_mouth_area_r": ["mouth_area_l", "mouth_area_r"],
    },
    "angle_compare": {
        "oepal0_iepal0_icpl_@ref": ["oepal0_iepal0_icpl", "@ref"],
        "iepal0_icpl_iepal1_@ref": ["iepal0_icpl_iepal1", "@ref"],
        "icpl_iepal1_oepal1_@ref": ["icpl_iepal1_oepal1", "@ref"],
        "iepal1_oepal1_ebp_@ref": ["iepal1_oepal1_ebp", "@ref"],
        "oepal1_ebp_npa1_@ref": ["oepal1_ebp_npa1", "@ref"],
        "npa1_ebp_oepar0_@ref": ["npa1_ebp_oepar0", "@ref"],
        "ebp_oepar0_iepar0_@ref": ["ebp_oepar0_iepar0", "@ref"],
        "oepar0_iepar0_icpr_@ref": ["oepar0_iepar0_icpr", "@ref"],
        "iepar0_icpr_iepar1_@ref": ["iepar0_icpr_iepar1", "@ref"],
        "icpr_iepar1_oepar1_@ref": ["icpr_iepar1_oepar1", "@ref"],
        "ebpal4_ebp_oepal1_@ref": ["ebpal4_ebp_oepal1", "@ref"],
        "ebpar0_ebp_oepar0_@ref": ["ebpar0_ebp_oepar0", "@ref"],
        "oepal0_cepal0_cepal1_@ref": ["oepal0_cepal0_cepal1", "@ref"],
        "oepal0_cepal1_oepal1_@ref": ["oepal0_cepal1_oepal1", "@ref"],
        "oepal1_npa0_npa2_@ref": ["oepal1_npa0_npa2", "@ref"],
        "npa0_ebp_npa1_@ref": ["npa0_ebp_npa1", "@ref"],
        "npa1_ebp_npa2_@ref": ["npa1_ebp_npa2", "@ref"],
        "npa0_npa1_mpa1_@ref": ["npa0_npa1_mpa1", "@ref"],
        "mpa1_npa1_npa2_@ref": ["mpa1_npa1_npa2", "@ref"],
        "npa0_npa2_oepar0_@ref": ["npa0_npa2_oepar0", "@ref"],
        "oepar0_npa2_cepar0_@ref": ["oepar0_npa2_cepar0", "@ref"],
        "cepal1_npa0_oepal1_@ref": ["cepal1_npa0_oepal1", "@ref"],
        "oepar0_cepar0_oepar1_@ref": ["oepar0_cepar0_oepar1", "@ref"],
        "cepar0_cepar1_oepar1_@ref": ["cepar0_cepar1_oepar1", "@ref"],
        "oepal0_mpa0_mpa1_@ref": ["oepal0_mpa0_mpa1", "@ref"],
        "mpa1_mpa2_oepar1_@ref": ["mpa1_mpa2_oepar1", "@ref"],
        "npa0_mpa0_mpa1_@ref": ["npa0_mpa0_mpa1", "@ref"],
        "mpa1_mpa2_npa2_@ref": ["mpa1_mpa2_npa2", "@ref"],
        "mpa0_mpa1_npa1_@ref": ["mpa0_mpa1_npa1", "@ref"],
        "npa1_mpa1_mpa2_@ref": ["npa1_mpa1_mpa2", "@ref"],
        "cipa0_mpa0_mpa1_@ref": ["cipa0_mpa0_mpa1", "@ref"],
        "mpa1_mpa2_cipa2_@ref": ["mpa1_mpa2_cipa2", "@ref"],
        "ebp_npa1_mpa1_@ref": ["ebp_npa1_mpa1", "@ref"],
        "npa1_mpa1_cipa1_@ref": ["npa1_mpa1_cipa1", "@ref"],
        "oepal0_iepal0_icpl_icpr_iepar1_oepar1": ["oepal0_iepal0_icpl", "icpr_iepar1_oepar1"],
        "iepal0_icpl_iepal1_iepar0_icpr_iepar1": ["iepal0_icpl_iepal1", "iepar0_icpr_iepar1"],
        "icpl_iepal1_oepal1_oepar0_iepar0_icpr": ["icpl_iepal1_oepal1", "oepar0_iepar0_icpr"],
        "iepal1_oepal1_ebp_ebp_oepar0_iepar0": ["iepal1_oepal1_ebp", "ebp_oepar0_iepar0"],
        "oepal1_ebp_npa1_npa1_ebp_oepar0": ["oepal1_ebp_npa1", "npa1_ebp_oepar0"],
        "ebpal4_ebp_oepal1_ebpar0_ebp_oepar0": ["ebpal4_ebp_oepal1", "ebpar0_ebp_oepar0"],
        "oepal0_cepal0_cepal1_cepar0_cepar1_oepar1": ["oepal0_cepal0_cepal1", "cepar0_cepar1_oepar1"],
        "oepal0_cepal1_oepal1_oepar0_cepar0_oepar1": ["oepal0_cepal1_oepal1", "oepar0_cepar0_oepar1"],
        "oepal1_npa0_npa2_npa0_npa2_oepar0": ["oepal1_npa0_npa2", "npa0_npa2_oepar0"],
        "cepal1_npa0_oepal1_oepar0_npa2_cepar0": ["cepal1_npa0_oepal1", "oepar0_npa2_cepar0"],
        "npa0_ebp_npa1_npa1_ebp_npa2": ["npa0_ebp_npa1", "npa1_ebp_npa2"],
        "npa0_npa1_mpa1_mpa1_npa1_npa2": ["npa0_npa1_mpa1", "mpa1_npa1_npa2"],
        "npa0_mpa0_mpa1_mpa1_mpa2_npa2": ["npa0_mpa0_mpa1", "mpa1_mpa2_npa2"],
        "mpa0_mpa1_npa1_npa1_mpa1_mpa2": ["mpa0_mpa1_npa1", "npa1_mpa1_mpa2"],
        "oepal0_mpa0_mpa1_mpa1_mpa2_oepar1_": ["oepal0_mpa0_mpa1", "mpa1_mpa2_oepar1_"],
        "cipa0_mpa0_mpa1_mpa1_mpa2_cipa2": ["cipa0_mpa0_mpa1", "mpa1_mpa2_cipa2"],
    },
}


def calculate_net(degree):
    point_heights_list_array = {
        "iris_center_point_l": [6],
        "iris_center_point_r": [6],
        "cheek_point_array_l": [-75, 15],
        "cheek_point_array_r": [15, -75],
        "inner_eye_point_array_l": [-45, 0],
        "inner_eye_point_array_r": [0, -45],
        "outer_eye_point_array_l": [-54, 6],
        "outer_eye_point_array_r": [6, -54],
        "eye_brow_point_array_l": [-54, 6, 36, 66, 51],
        "eye_brow_point_array_r": [51, 66, 36, 6, -54],
        "eye_bridge_point": [0],
        "nose_point_array": [30, 120, 30],
        "mouth_point_array": [0, 90, 0],
        "chin_point_array": [15, 36, 15],
    }
    points_0_degree = {
        "iris_center_point_l": [[265, 100]],
        "iris_center_point_r": [[265, 280]],
        "cheek_point_array_l": [[310, 44], [340, 94]],
        "cheek_point_array_r": [[340, 286], [310, 335]],
        "inner_eye_point_array_l": [[265, 64], [265, 136]],
        "inner_eye_point_array_r": [[265, 244], [265, 316]],
        "outer_eye_point_array_l": [[265, 48], [265, 153]],
        "outer_eye_point_array_r": [[265, 227], [265, 332]],
        "eye_brow_point_array_l": [[235, 47], [220, 74], [214, 100], [220, 121], [235, 143]],
        "eye_brow_point_array_r": [[235, 237], [220, 259], [214, 280], [220, 306], [235, 332]],
        "eye_bridge_point": [[265, 190]],
        "nose_point_array": [[370, 136], [355, 190], [370, 244]],
        "mouth_point_array": [[445, 100], [445, 190], [445, 280]],
        "chin_point_array": [[535, 154], [505, 190], [535, 226]],
    }  # raw_points_0_degree
    for a, b in zip(point_heights_list_array, points_0_degree):
        point_heights_array = point_heights_list_array[a]
        point_array = points_0_degree[b]
        for z, p in zip(point_heights_array, point_array):
            already_angle = math.atan(z / p[1])
            result_angle = already_angle + (2*math.pi * (degree/360))
            p[1] = int(math.cos(result_angle) * sqrt(z ** 2 + p[1] ** 2))
    face_net = {
        "point_heights": {
            "iris_center_point_l": [6],
            "iris_center_point_r": [6],
            "cheek_point_array_l": [-75, 15],
            "cheek_point_array_r": [15, -75],
            "inner_eye_point_array_l": [-45, 0],
            "inner_eye_point_array_r": [0, -45],
            "outer_eye_point_array_l": [-54, 6],
            "outer_eye_point_array_r": [6, -54],
            "eye_brow_point_array_l": [-54, 6, 36, 66, 51],
            "eye_brow_point_array_r": [51, 66, 36, 6, -54],
            "eye_bridge_point": [0],
            "nose_point_array": [30, 120, 30],
            "mouth_point_array": [0, 90, 0],
            "chin_point_array": [15, 36, 15],
        },
        "points": points_0_degree,
        "lines": {
            "ebpal0_ebpal1": [points_0_degree["eye_brow_point_array_l"][0], points_0_degree["eye_brow_point_array_l"][1]],
            "ebpal1_ebpal2": [points_0_degree["eye_brow_point_array_l"][1], points_0_degree["eye_brow_point_array_l"][2]],
            "ebpal2_ebpal3": [points_0_degree["eye_brow_point_array_l"][2], points_0_degree["eye_brow_point_array_l"][3]],
            "ebpal3_ebpal4": [points_0_degree["eye_brow_point_array_l"][3], points_0_degree["eye_brow_point_array_l"][4]],
            "ebpar0_ebpar1": [points_0_degree["eye_brow_point_array_r"][0], points_0_degree["eye_brow_point_array_r"][1]],
            "ebpar1_ebpar2": [points_0_degree["eye_brow_point_array_r"][1], points_0_degree["eye_brow_point_array_r"][2]],
            "ebpar2_ebpar3": [points_0_degree["eye_brow_point_array_r"][2], points_0_degree["eye_brow_point_array_r"][3]],
            "ebpar3_ebpar4": [points_0_degree["eye_brow_point_array_r"][3], points_0_degree["eye_brow_point_array_r"][4]],
            "oepal0_oepal1": [points_0_degree["outer_eye_point_array_l"][0], points_0_degree["outer_eye_point_array_l"][1]],
            "oepar0_oepar1": [points_0_degree["outer_eye_point_array_r"][0], points_0_degree["outer_eye_point_array_r"][1]],
            "iepal0_iepal1": [points_0_degree["inner_eye_point_array_l"][0], points_0_degree["inner_eye_point_array_l"][1]],
            "iepar0_iepar1": [points_0_degree["inner_eye_point_array_r"][0], points_0_degree["inner_eye_point_array_r"][1]],
            "oepal0_iepal0": [points_0_degree["outer_eye_point_array_l"][0], points_0_degree["inner_eye_point_array_l"][0]],
            "iepar1_oepar1": [points_0_degree["inner_eye_point_array_r"][1], points_0_degree["outer_eye_point_array_r"][1]],
            "iepal1_oepal1": [points_0_degree["inner_eye_point_array_l"][1], points_0_degree["outer_eye_point_array_l"][1]],
            "oepar0_iepar0": [points_0_degree["outer_eye_point_array_r"][0], points_0_degree["inner_eye_point_array_r"][0]],
            "iepal0_icpl": [points_0_degree["inner_eye_point_array_l"][0], points_0_degree["iris_center_point_l"][0]],
            "icpl_iepal1": [points_0_degree["iris_center_point_l"][0], points_0_degree["inner_eye_point_array_l"][1]],
            "iepar0_icpr": [points_0_degree["inner_eye_point_array_r"][0], points_0_degree["iris_center_point_r"][0]],
            "icpr_iepar1": [points_0_degree["iris_center_point_r"][0], points_0_degree["inner_eye_point_array_r"][1]],
            "oepal1_ebp": [points_0_degree["outer_eye_point_array_l"][1], points_0_degree["eye_bridge_point"][0]],
            "ebp_oepar0": [points_0_degree["eye_bridge_point"][0], points_0_degree["outer_eye_point_array_r"][0]],
            "ebpal4_ebp": [points_0_degree["eye_brow_point_array_l"][4], points_0_degree["eye_bridge_point"][0]],
            "ebp_ebpar0": [points_0_degree["eye_bridge_point"][0], points_0_degree["eye_brow_point_array_r"][0]],
            "oepal0_cepal0": [points_0_degree["outer_eye_point_array_l"][0], points_0_degree["cheek_point_array_l"][0]],
            "oepar1_cepar1": [points_0_degree["outer_eye_point_array_r"][1], points_0_degree["cheek_point_array_r"][1]],
            "oepal0_cepal1": [points_0_degree["outer_eye_point_array_l"][0], points_0_degree["cheek_point_array_l"][1]],
            "oepar1_cepar0": [points_0_degree["outer_eye_point_array_r"][1], points_0_degree["cheek_point_array_r"][0]],
            "cepal0_cepal1": [points_0_degree["cheek_point_array_l"][0], points_0_degree["cheek_point_array_l"][1]],
            "cepar0_cepar1": [points_0_degree["cheek_point_array_r"][0], points_0_degree["cheek_point_array_r"][1]],
            "cepal1_oepal1": [points_0_degree["cheek_point_array_l"][1], points_0_degree["outer_eye_point_array_l"][1]],
            "oepar0_cepar0": [points_0_degree["outer_eye_point_array_r"][0], points_0_degree["cheek_point_array_r"][0]],
            "cepal1_npa0": [points_0_degree["cheek_point_array_l"][1], points_0_degree["nose_point_array"][0]],
            "npa2_cepar0": [points_0_degree["nose_point_array"][2], points_0_degree["cheek_point_array_r"][0]],
            "npa0_oepal1": [points_0_degree["nose_point_array"][0], points_0_degree["outer_eye_point_array_l"][1]],
            "oepar0_npa2": [points_0_degree["outer_eye_point_array_r"][0], points_0_degree["nose_point_array"][2]],
            "npa0_ebp": [points_0_degree["nose_point_array"][0], points_0_degree["eye_bridge_point"][0]],
            "ebp_npa2": [points_0_degree["eye_bridge_point"][0], points_0_degree["nose_point_array"][2]],
            "npa0_npa2": [points_0_degree["nose_point_array"][0], points_0_degree["nose_point_array"][2]],
            "ebp_npa1": [points_0_degree["eye_bridge_point"][0], points_0_degree["nose_point_array"][1]],
            "npa1_mpa1": [points_0_degree["nose_point_array"][1], points_0_degree["mouth_point_array"][1]],
            "mpa0_npa0": [points_0_degree["mouth_point_array"][0], points_0_degree["nose_point_array"][0]],
            "npa2_mpa2": [points_0_degree["nose_point_array"][2], points_0_degree["mouth_point_array"][2]],
            "mpa0_mpa1": [points_0_degree["mouth_point_array"][0], points_0_degree["mouth_point_array"][1]],
            "mpa1_mpa2": [points_0_degree["mouth_point_array"][1], points_0_degree["mouth_point_array"][2]],
            "npa0_cipa0": [points_0_degree["nose_point_array"][0], points_0_degree["chin_point_array"][0]],
            "npa0_npa1": [points_0_degree["nose_point_array"][0], points_0_degree["nose_point_array"][1]],
            "npa1_npa2": [points_0_degree["nose_point_array"][1], points_0_degree["nose_point_array"][2]],
            "cipa2_npa2": [points_0_degree["chin_point_array"][2], points_0_degree["nose_point_array"][2]],
            "mpa1_cipa1": [points_0_degree["mouth_point_array"][1], points_0_degree["chin_point_array"][1]],
            "cipa0_cipa1": [points_0_degree["chin_point_array"][0], points_0_degree["chin_point_array"][1]],
            "cipa1_cipa2": [points_0_degree["chin_point_array"][1], points_0_degree["chin_point_array"][2]],
            "cipa0_cipa2": [points_0_degree["chin_point_array"][0], points_0_degree["chin_point_array"][2]],
            "cepal1_mpa0": [points_0_degree["cheek_point_array_l"][1], points_0_degree["mouth_point_array"][0]],
            "mpa2_cepar0": [points_0_degree["mouth_point_array"][2], points_0_degree["cheek_point_array_r"][0]],
            "oepal0_mpa0": [points_0_degree["outer_eye_point_array_l"][0], points_0_degree["mouth_point_array"][0]],
            "mpa2_oepar1": [points_0_degree["mouth_point_array"][2], points_0_degree["outer_eye_point_array_r"][1]],
            "mpa0_cipa0": [points_0_degree["mouth_point_array"][0], points_0_degree["chin_point_array"][0]],
            "cipa2_mpa2": [points_0_degree["chin_point_array"][2], points_0_degree["mouth_point_array"][2]],
        },
        "areas": {
            "iris_area_l": [],
            "iris_area_r": [],
            "inner_eye_area_l": [],
            "inner_eye_area_r": [],
            "outer_eye_area_l": [],
            "outer_eye_area_r": [],
            "mouth_area": [],
        },
        "angles": {
            "oepal0_iepal0_icpl": [points_0_degree["outer_eye_point_array_l"][0],
                                   points_0_degree["inner_eye_point_array_l"][0],
                                   points_0_degree["iris_center_point_l"][0]],
            "iepal0_icpl_iepal1": [points_0_degree["inner_eye_point_array_l"][0], points_0_degree["iris_center_point_l"][0],
                                   points_0_degree["inner_eye_point_array_l"][1]],
            "icpl_iepal1_oepal1": [points_0_degree["iris_center_point_l"][0], points_0_degree["inner_eye_point_array_l"][1],
                                   points_0_degree["outer_eye_point_array_l"][1]],
            "iepal1_oepal1_ebp": [points_0_degree["inner_eye_point_array_l"][1],
                                  points_0_degree["outer_eye_point_array_l"][1], points_0_degree["eye_bridge_point"][0]],
            "oepal1_ebp_npa1": [points_0_degree["outer_eye_point_array_l"][1], points_0_degree["eye_bridge_point"][0],
                                points_0_degree["nose_point_array"][1]],
            "npa1_ebp_oepar0": [points_0_degree["nose_point_array"][1], points_0_degree["eye_bridge_point"][0],
                                points_0_degree["outer_eye_point_array_r"][0]],
            "ebp_oepar0_iepar0": [points_0_degree["eye_bridge_point"][0], points_0_degree["outer_eye_point_array_r"][0],
                                  points_0_degree["inner_eye_point_array_r"][0]],
            "oepar0_iepar0_icpr": [points_0_degree["outer_eye_point_array_r"][0],
                                   points_0_degree["inner_eye_point_array_r"][0],
                                   points_0_degree["iris_center_point_r"][0]],
            "iepar0_icpr_iepar1": [points_0_degree["inner_eye_point_array_r"][0], points_0_degree["iris_center_point_r"][0],
                                   points_0_degree["inner_eye_point_array_r"][1]],
            "icpr_iepar1_oepar1": [points_0_degree["iris_center_point_r"][0], points_0_degree["inner_eye_point_array_r"][1],
                                   points_0_degree["outer_eye_point_array_r"][1]],
            "ebpal4_ebp_oepal1": [points_0_degree["eye_brow_point_array_l"][4], points_0_degree["eye_bridge_point"][0],
                                  points_0_degree["outer_eye_point_array_l"][1]],
            "ebpar0_ebp_oepar0": [points_0_degree["eye_brow_point_array_l"][0], points_0_degree["eye_bridge_point"][0],
                                  points_0_degree["outer_eye_point_array_r"][0]],
            "oepal0_cepal0_cepal1": [points_0_degree["outer_eye_point_array_l"][0],
                                     points_0_degree["cheek_point_array_l"][0], points_0_degree["cheek_point_array_l"][1]],
            "oepal0_cepal1_oepal1": [points_0_degree["outer_eye_point_array_l"][0],
                                     points_0_degree["cheek_point_array_l"][1],
                                     points_0_degree["outer_eye_point_array_l"][1]],
            "oepal1_npa0_npa2": [points_0_degree["outer_eye_point_array_l"][1], points_0_degree["nose_point_array"][0],
                                 points_0_degree["nose_point_array"][2]],
            "npa0_ebp_npa1": [points_0_degree["nose_point_array"][0], points_0_degree["eye_bridge_point"][0],
                              points_0_degree["nose_point_array"][1]],
            "npa1_ebp_npa2": [points_0_degree["nose_point_array"][1], points_0_degree["eye_bridge_point"][0],
                              points_0_degree["nose_point_array"][2]],
            "npa0_npa1_mpa1": [points_0_degree["nose_point_array"][0], points_0_degree["nose_point_array"][1],
                               points_0_degree["mouth_point_array"][1]],
            "mpa1_npa1_npa2": [points_0_degree["mouth_point_array"][1], points_0_degree["nose_point_array"][1],
                               points_0_degree["nose_point_array"][2]],
            "npa0_npa2_oepar0": [points_0_degree["nose_point_array"][0], points_0_degree["nose_point_array"][2],
                                 points_0_degree["outer_eye_point_array_r"][0]],
            "oepar0_npa2_cepar0": [points_0_degree["outer_eye_point_array_r"][0], points_0_degree["nose_point_array"][2],
                                   points_0_degree["cheek_point_array_r"][0]],
            "cepal1_npa0_oepal1": [points_0_degree["cheek_point_array_l"][1], points_0_degree["nose_point_array"][0],
                                   points_0_degree["outer_eye_point_array_l"][1]],
            "oepar0_cepar0_oepar1": [points_0_degree["outer_eye_point_array_r"][0],
                                     points_0_degree["cheek_point_array_r"][0],
                                     points_0_degree["outer_eye_point_array_r"][1]],
            "cepar0_cepar1_oepar1": [points_0_degree["cheek_point_array_r"][0], points_0_degree["cheek_point_array_r"][1],
                                     points_0_degree["outer_eye_point_array_r"][1]],
            "oepal0_mpa0_mpa1": [points_0_degree["outer_eye_point_array_l"][0], points_0_degree["mouth_point_array"][0],
                                 points_0_degree["mouth_point_array"][1]],
            "mpa1_mpa2_oepar1": [points_0_degree["mouth_point_array"][1], points_0_degree["mouth_point_array"][2],
                                 points_0_degree["outer_eye_point_array_r"][1]],
            "npa0_mpa0_mpa1": [points_0_degree["nose_point_array"][0], points_0_degree["mouth_point_array"][0],
                               points_0_degree["mouth_point_array"][1]],
            "mpa1_mpa2_npa2": [points_0_degree["mouth_point_array"][1], points_0_degree["mouth_point_array"][2],
                               points_0_degree["nose_point_array"][2]],
            "mpa0_mpa1_npa1": [points_0_degree["mouth_point_array"][0], points_0_degree["mouth_point_array"][1],
                               points_0_degree["nose_point_array"][1]],
            "npa1_mpa1_mpa2": [points_0_degree["nose_point_array"][1], points_0_degree["mouth_point_array"][1],
                               points_0_degree["mouth_point_array"][2]],
            "cipa0_mpa0_mpa1": [points_0_degree["chin_point_array"][0], points_0_degree["mouth_point_array"][0],
                                points_0_degree["mouth_point_array"][1]],
            "mpa1_mpa2_cipa2": [points_0_degree["mouth_point_array"][1], points_0_degree["mouth_point_array"][2],
                                points_0_degree["chin_point_array"][2]],
            "ebp_npa1_mpa1": [points_0_degree["eye_bridge_point"][0], points_0_degree["nose_point_array"][1],
                              points_0_degree["mouth_point_array"][1]],
            "npa1_mpa1_cipa1": [points_0_degree["nose_point_array"][1], points_0_degree["mouth_point_array"][1],
                                points_0_degree["chin_point_array"][1]],
        },
        "line_compare": {
            "oepal0_oepal1@ref": ["oepal0_oepal1", "@ref"],
            "oepar0_oepar1@ref": ["oepar0_oepar1", "@ref"],
            "iepal0_iepal1@ref": ["iepal0_iepal1", "@ref"],
            "iepar0_iepar1@ref": ["iepar0_iepar1", "@ref"],
            "oepal0_iepal0@ref": ["oepal0_iepal0", "@ref"],
            "iepar1_oepar1@ref": ["iepar1_oepar1", "@ref"],
            "iepal1_oepal1@ref": ["iepal1_oepal1", "@ref"],
            "oepar0_iepar0@ref": ["oepar0_iepar0", "@ref"],
            "iepal0_icpl@ref": ["iepal0_icpl", "@ref"],
            "icpl_iepal1@ref": ["icpl_iepal1", "@ref"],
            "iepar0_icpr@ref": ["iepar0_icpr", "@ref"],
            "icpr_iepar1@ref": ["icpr_iepar1", "@ref"],
            "oepal1_ebp@ref": ["oepal1_ebp", "@ref"],
            "ebp_oepar0@ref": ["ebp_oepar0", "@ref"],
            "ebpal4_ebp@ref": ["ebpal4_ebp", "@ref"],
            "ebp_ebpar0@ref": ["ebp_ebpar0", "@ref"],
            "oepal0_cepal0@ref": ["oepal0_cepal0", "@ref"],
            "oepar1_cepar1@ref": ["oepar1_cepar1", "@ref"],
            "oepal0_cepal1@ref": ["oepal0_cepal1", "@ref"],
            "oepar1_cepar0@ref": ["oepar1_cepar0", "@ref"],
            "cepal0_cepal1@ref": ["cepal0_cepal1", "@ref"],
            "cepar0_cepar1@ref": ["cepar0_cepar1", "@ref"],
            "cepal1_oepal1@ref": ["cepal1_oepal1", "@ref"],
            "oepar0_cepar0@ref": ["oepar0_cepar0", "@ref"],
            "cepal1_npa0@ref": ["cepal1_npa0", "@ref"],
            "npa2_cepar0@ref": ["npa2_cepar0", "@ref"],
            "npa0_oepal1@ref": ["npa0_oepal1", "@ref"],
            "oepar0_npa2@ref": ["oepar0_npa2", "@ref"],
            "npa0_ebp@ref": ["npa0_ebp", "@ref"],
            "ebp_npa2@ref": ["ebp_npa2", "@ref"],
            "npa0_npa2@ref": ["npa0_npa2", "@ref"],
            "ebp_npa1@ref": ["ebp_npa1", "@ref"],
            "npa1_mpa1@ref": ["npa1_mpa1", "@ref"],
            "mpa0_npa0@ref": ["mpa0_npa0", "@ref"],
            "npa2_mpa2@ref": ["npa2_mpa2", "@ref"],
            "mpa0_mpa1@ref": ["mpa0_mpa1", "@ref"],
            "mpa1_mpa2@ref": ["mpa1_mpa2", "@ref"],
            "npa0_cipa0@ref": ["npa0_cipa0", "@ref"],
            "cipa2_npa2@ref": ["cipa2_npa2", "@ref"],
            "mpa1_cipa1@ref": ["mpa1_cipa1", "@ref"],
            "cipa0_cipa1@ref": ["cipa0_cipa1", "@ref"],
            "cipa1_cipa2@ref": ["cipa1_cipa2", "@ref"],
            "cipa0_cipa2@ref": ["cipa0_cipa2", "@ref"],
            "cepal1_mpa0@ref": ["cepal1_mpa0", "@ref"],
            "mpa2_cepar0@ref": ["mpa2_cepar0", "@ref"],
            "oepal0_mpa0@ref": ["oepal0_mpa0", "@ref"],
            "mpa2_oepar1@ref": ["mpa2_oepar1", "@ref"],
            "oepal0_oepal1_oepar0_oepar1": ["oepal0_oepal1", "oepar0_oepar1"],
            "iepal0_iepal1_iepar0_iepar1": ["iepal0_iepal1", "iepar0_iepar1"],
            "oepal0_iepal0_iepar1_oepar1": ["oepal0_iepal0", "iepar1_oepar1"],
            "iepal1_oepal1_oepar0_iepar0": ["iepal1_oepal1", "oepar0_iepar0"],
            "iepal0_icpl_iepar0_icpr": ["iepal0_icpl", "iepar0_icpr"],
            "icpl_iepal1_icpr_iepar1": ["icpl_iepal1", "icpr_iepar1"],
            "oepal1_ebp_ebp_oepar0": ["oepal1_ebp", "ebp_oepar0"],
            "ebpal4_ebp_ebp_ebpar0": ["ebpal4_ebp", "ebp_ebpar0"],
            "oepal0_cepal0_oepar1_cepar1": ["oepal0_cepal0", "oepar1_cepar1"],
            "oepal0_cepal1_oepar1_cepar0": ["oepal0_cepal1", "oepar1_cepar0"],
            "cepal1_oepal1_oepar0_cepar0": ["cepal1_oepal1", "oepar0_cepar0"],
            "cepal1_npa0_npa2_cepar0": ["cepal1_npa0", "npa2_cepar0"],
            "npa0_oepal1_oepar0_npa2": ["npa0_oepal1", "oepar0_npa2"],
            "npa0_ebp_ebp_npa2": ["npa0_ebp", "ebp_npa2"],
            "ebp_npa1_npa1_mpa1": ["ebp_npa1", "npa1_mpa1"],
            "npa0_npa1_npa1_npa2": ["npa0_npa1", "npa1_npa2"],
            "mpa0_npa0_npa2_mpa2": ["mpa0_npa0", "npa2_mpa2"],
            "mpa0_mpa1_mpa1_mpa2": ["mpa0_mpa1", "mpa1_mpa2"],
            "npa0_cipa0_cipa2_npa2": ["npa0_cipa0", "cipa2_npa2"],
            "mpa0_cipa0_cipa2_mpa2": ["mpa0_cipa0", "cipa2_mpa2"],
            "npa1_mpa1_mpa1_cipa1": ["npa1_mpa1", "mpa1_cipa1"],
            "cipa0_cipa1_cipa1_cipa2": ["cipa0_cipa1", "cipa1_cipa2"],
            "cepal1_mpa0_mpa2_cepar0": ["cepal1_mpa0", "mpa2_cepar0"],
            "oepal0_mpa0_mpa2_oepar1": ["oepal0_mpa0", "mpa2_oepar1"],
        },
        "area_compare": {
            "iris_area_l_@ref": ["iris_area_l", "@ref"],
            "iris_area_r_@ref": ["iris_area_r", "@ref"],
            "inner_eye_area_l_@ref": ["inner_eye_area_l", "@ref"],
            "inner_eye_area_r_@ref": ["inner_eye_area_r", "@ref"],
            "outer_eye_area_l_@ref": ["outer_eye_area_l", "@ref"],
            "outer_eye_area_r_@ref": ["outer_eye_area_r", "@ref"],
            "mouth_area_@ref": ["mouth_area", "@ref"],
            "iris_area_l_iris_are_r": ["iris_area_l", "iris_are_r"],
            "inner_eye_area_l_inner_eye_area_r": ["inner_eye_area_l", "inner_eye_area_r"],
            "outer_eye_area_l_outer_eye_area_r": ["outer_eye_area_l", "outer_eye_area_r"],
            "mouth_area_l_mouth_area_r": ["mouth_area_l", "mouth_area_r"],
        },
        "angle_compare": {
            "oepal0_iepal0_icpl_@ref": ["oepal0_iepal0_icpl", "@ref"],
            "iepal0_icpl_iepal1_@ref": ["iepal0_icpl_iepal1", "@ref"],
            "icpl_iepal1_oepal1_@ref": ["icpl_iepal1_oepal1", "@ref"],
            "iepal1_oepal1_ebp_@ref": ["iepal1_oepal1_ebp", "@ref"],
            "oepal1_ebp_npa1_@ref": ["oepal1_ebp_npa1", "@ref"],
            "npa1_ebp_oepar0_@ref": ["npa1_ebp_oepar0", "@ref"],
            "ebp_oepar0_iepar0_@ref": ["ebp_oepar0_iepar0", "@ref"],
            "oepar0_iepar0_icpr_@ref": ["oepar0_iepar0_icpr", "@ref"],
            "iepar0_icpr_iepar1_@ref": ["iepar0_icpr_iepar1", "@ref"],
            "icpr_iepar1_oepar1_@ref": ["icpr_iepar1_oepar1", "@ref"],
            "ebpal4_ebp_oepal1_@ref": ["ebpal4_ebp_oepal1", "@ref"],
            "ebpar0_ebp_oepar0_@ref": ["ebpar0_ebp_oepar0", "@ref"],
            "oepal0_cepal0_cepal1_@ref": ["oepal0_cepal0_cepal1", "@ref"],
            "oepal0_cepal1_oepal1_@ref": ["oepal0_cepal1_oepal1", "@ref"],
            "oepal1_npa0_npa2_@ref": ["oepal1_npa0_npa2", "@ref"],
            "npa0_ebp_npa1_@ref": ["npa0_ebp_npa1", "@ref"],
            "npa1_ebp_npa2_@ref": ["npa1_ebp_npa2", "@ref"],
            "npa0_npa1_mpa1_@ref": ["npa0_npa1_mpa1", "@ref"],
            "mpa1_npa1_npa2_@ref": ["mpa1_npa1_npa2", "@ref"],
            "npa0_npa2_oepar0_@ref": ["npa0_npa2_oepar0", "@ref"],
            "oepar0_npa2_cepar0_@ref": ["oepar0_npa2_cepar0", "@ref"],
            "cepal1_npa0_oepal1_@ref": ["cepal1_npa0_oepal1", "@ref"],
            "oepar0_cepar0_oepar1_@ref": ["oepar0_cepar0_oepar1", "@ref"],
            "cepar0_cepar1_oepar1_@ref": ["cepar0_cepar1_oepar1", "@ref"],
            "oepal0_mpa0_mpa1_@ref": ["oepal0_mpa0_mpa1", "@ref"],
            "mpa1_mpa2_oepar1_@ref": ["mpa1_mpa2_oepar1", "@ref"],
            "npa0_mpa0_mpa1_@ref": ["npa0_mpa0_mpa1", "@ref"],
            "mpa1_mpa2_npa2_@ref": ["mpa1_mpa2_npa2", "@ref"],
            "mpa0_mpa1_npa1_@ref": ["mpa0_mpa1_npa1", "@ref"],
            "npa1_mpa1_mpa2_@ref": ["npa1_mpa1_mpa2", "@ref"],
            "cipa0_mpa0_mpa1_@ref": ["cipa0_mpa0_mpa1", "@ref"],
            "mpa1_mpa2_cipa2_@ref": ["mpa1_mpa2_cipa2", "@ref"],
            "ebp_npa1_mpa1_@ref": ["ebp_npa1_mpa1", "@ref"],
            "npa1_mpa1_cipa1_@ref": ["npa1_mpa1_cipa1", "@ref"],
            "oepal0_iepal0_icpl_icpr_iepar1_oepar1": ["oepal0_iepal0_icpl", "icpr_iepar1_oepar1"],
            "iepal0_icpl_iepal1_iepar0_icpr_iepar1": ["iepal0_icpl_iepal1", "iepar0_icpr_iepar1"],
            "icpl_iepal1_oepal1_oepar0_iepar0_icpr": ["icpl_iepal1_oepal1", "oepar0_iepar0_icpr"],
            "iepal1_oepal1_ebp_ebp_oepar0_iepar0": ["iepal1_oepal1_ebp", "ebp_oepar0_iepar0"],
            "oepal1_ebp_npa1_npa1_ebp_oepar0": ["oepal1_ebp_npa1", "npa1_ebp_oepar0"],
            "ebpal4_ebp_oepal1_ebpar0_ebp_oepar0": ["ebpal4_ebp_oepal1", "ebpar0_ebp_oepar0"],
            "oepal0_cepal0_cepal1_cepar0_cepar1_oepar1": ["oepal0_cepal0_cepal1", "cepar0_cepar1_oepar1"],
            "oepal0_cepal1_oepal1_oepar0_cepar0_oepar1": ["oepal0_cepal1_oepal1", "oepar0_cepar0_oepar1"],
            "oepal1_npa0_npa2_npa0_npa2_oepar0": ["oepal1_npa0_npa2", "npa0_npa2_oepar0"],
            "cepal1_npa0_oepal1_oepar0_npa2_cepar0": ["cepal1_npa0_oepal1", "oepar0_npa2_cepar0"],
            "npa0_ebp_npa1_npa1_ebp_npa2": ["npa0_ebp_npa1", "npa1_ebp_npa2"],
            "npa0_npa1_mpa1_mpa1_npa1_npa2": ["npa0_npa1_mpa1", "mpa1_npa1_npa2"],
            "npa0_mpa0_mpa1_mpa1_mpa2_npa2": ["npa0_mpa0_mpa1", "mpa1_mpa2_npa2"],
            "mpa0_mpa1_npa1_npa1_mpa1_mpa2": ["mpa0_mpa1_npa1", "npa1_mpa1_mpa2"],
            "oepal0_mpa0_mpa1_mpa1_mpa2_oepar1_": ["oepal0_mpa0_mpa1", "mpa1_mpa2_oepar1_"],
            "cipa0_mpa0_mpa1_mpa1_mpa2_cipa2": ["cipa0_mpa0_mpa1", "mpa1_mpa2_cipa2"],
        },
    }  # raw_face_net_0_degree
    face_net = calculate_areas(degree, face_net)
    return face_net


def draw_net(face_net):
    face_net_picture = np.zeros(BLACK_PICTURE_0_DEGREE.shape)
    # face_net = calculate_areas()

    for area in face_net["areas"]:
        area_picture = face_net["areas"][area]
        face_net_picture = face_net_picture + area_picture

    for line in face_net["lines"]:
        line = face_net["lines"][line]
        vector = tuple(np.subtract(line[0], line[1]))
        vector_length = round(sqrt(vector[0] ** 2 + vector[1] ** 2))
        for i in range(0, vector_length):
            relative_length = i / vector_length
            point = np.subtract(line[0], (int(round(relative_length * vector[0])), int(round(relative_length * vector[1]))))
            face_net_picture[point[0]][point[1]] = 20
    for point_entry in face_net["points"]:
        points = face_net["points"][point_entry]
        for point in points:
            face_net_picture[point[0]][point[1]] = 40
    io.imshow(face_net_picture)
    plt.show()
    return face_net_picture


def calculate_areas(degree, face_net):
    face_net["areas"]["iris_area_l"] = iris_area("left", degree)
    face_net["areas"]["iris_area_r"] = iris_area("right", degree)
    face_net["areas"]["inner_eye_area_l"] = inner_eye_area("left", degree)
    face_net["areas"]["inner_eye_area_r"] = inner_eye_area("right", degree)
    face_net["areas"]["outer_eye_area_l"] = outer_eye_area("left", degree)
    face_net["areas"]["outer_eye_area_r"] = outer_eye_area("right", degree)
    face_net["areas"]["mouth_area"] = mouth_area(degree)
    return face_net


def iris_area(place, degree):
    """Alles innerhalb des Kreises mit dem Radius 30 und dem Mittelpunkt: iris_center_point_l/r
       h = ((-2/75)*x**2 + (4/5)*x)
       """
    iris_picture = np.zeros(BLACK_PICTURE_0_DEGREE.shape)
    iris_center_point = raw_points_0_degree["iris_center_point_l"][0] if place == "left" else raw_points_0_degree["iris_center_point_r"][0]
    for y in range(iris_center_point[0] - 15, iris_center_point[0] + 15):  # iris_picture.shape[0]
        for x in range(iris_center_point[1] - 15, iris_center_point[1] + 15):
            iris_picture[y][x] = 10 if 15 >= sqrt((y - iris_center_point[0]) ** 2 + (x - iris_center_point[1]) ** 2) else 0
    if not degree == 0:
        degree_iris_picture = np.zeros(BLACK_PICTURE_0_DEGREE.shape)
        for y in range(iris_center_point[0] - 15, iris_center_point[0] + 15):
            for x in range(iris_center_point[1] - 15, iris_center_point[1] + 15):
                z = ((-2 / 75) * (x - iris_center_point[1] + 15)**2 + (4 / 5) * (x - iris_center_point[1] + 15))
                already_angle = math.atan(z / x)
                result_angle = already_angle + (2*math.pi * (degree/360))
                new_x = round(math.cos(result_angle) * sqrt(z ** 2 + x ** 2))
                degree_iris_picture[y][new_x] = iris_picture[y][x]
                if degree_iris_picture[y][new_x-2] == degree_iris_picture[y][new_x] == 10:
                    degree_iris_picture[y][new_x-1] = 10
        return degree_iris_picture
    return iris_picture


def inner_eye_area(place, degree):
    """B=(30/11654297)*x^3 + (-129915/11654297)*x^2 + (256500/314981)*x
       -B=-(30/11654297)*x^3 - (-129915/11654297)*x^2 - (256500/314981)*x
       h_r = ((-4077/9884180) * x**3 + (129189/4942090) * x**2 - (64209/267140) * x)
       h_l = ((2181/4942090) * x**3 + (-172752/2471045) * x**2 + (448917/133570) * x - 45)
       """
    area_place = raw_points_0_degree["inner_eye_point_array_l"] if place == "left" else raw_points_0_degree["inner_eye_point_array_r"]
    inner_eye_picture = np.zeros(BLACK_PICTURE_0_DEGREE.shape)
    for y in range(-15, 15):
        for x in range(0, area_place[1][1] - area_place[0][1]):
            inner_eye_picture[area_place[0][0] + y][area_place[0][1] + x] = 10 if ((30 / 11654297) * x ** 3 + (-129915 / 11654297) * x ** 2 + (256500 / 314981) * x) >= y >= (-(30 / 11654297) * x ** 3 - (-129915 / 11654297) * x ** 2 - (256500 / 314981) * x) else 0
    if not degree == 0:
        degree_inner_eye_picture = np.zeros(BLACK_PICTURE_0_DEGREE.shape)
        for y in range(area_place[0][0] - 15, area_place[0][0] + 15):
            for x in range(area_place[0][1], area_place[1][1]):
                z = ((2181 / 4942090) * (x - area_place[0][1]) ** 3 + (-172752 / 2471045) * (x - area_place[0][1]) ** 2 + (448917 / 133570) * (x - area_place[0][1]) - 45) if place == "left" else ((-4077 / 9884180) * (x - area_place[0][1]) ** 3 + (129189 / 4942090) * (x - area_place[0][1]) ** 2 - (64209 / 267140) * (x - area_place[0][1]))
                already_angle = math.atan(z / x)
                result_angle = already_angle + (2*math.pi * (degree/360))
                new_x = round(math.cos(result_angle) * sqrt(z ** 2 + x ** 2))
                degree_inner_eye_picture[y][new_x] = inner_eye_picture[y][x]
                if degree_inner_eye_picture[y][new_x-2] == degree_inner_eye_picture[y][new_x] == 10:
                    degree_inner_eye_picture[y][new_x-1] = 10
        return degree_inner_eye_picture
    return inner_eye_picture


def outer_eye_area(place, degree):
    """B_o=((42 / 46160297)*x**3 + (340641 / 46160297)*x**2 + (-687960 / 870949)*x)
       -B_u=((-60 / 46160297)*x**3 + (-486630 / 46160297)*x**2 + (982800 / 870949)*x)
       h_r = ((-5387 / 26584376)*x**3 + (501259 / 26584376)*x**2 + (-81073 / 255619)*x + 6)
       h_l = ((5639 / 26584376)*x**3 + (-1235335 / 26584376)*x**2 + (795493 / 255619)*x - 54)
       """
    area_place = raw_points_0_degree["outer_eye_point_array_l"] if place == "left" else raw_points_0_degree["outer_eye_point_array_r"]
    outer_eye_picture = np.zeros(BLACK_PICTURE_0_DEGREE.shape)
    for y in range(-21, 30):
        for x in range(0, area_place[1][1] - area_place[0][1]):
            outer_eye_picture[area_place[0][0] + y][area_place[0][1] + x] = 10 if ((-60 / 46160297)*x**3 + (-486630 / 46160297)*x**2 + (982800 / 870949)*x) >= y >= ((42 / 46160297)*x**3 + (340641 / 46160297)*x**2 + (-687960 / 870949)*x) else 0
    if not degree == 0:
        degree_outer_eye_picture = np.zeros(BLACK_PICTURE_0_DEGREE.shape)
        for y in range(area_place[0][0] - 21, area_place[0][0] + 30):
            for x in range(area_place[0][1], area_place[1][1]):
                z = ((5639 / 26584376)*(x - area_place[0][1])**3 + (-1235335 / 26584376)*(x - area_place[0][1])**2 + (795493 / 255619)*(x - area_place[0][1]) - 54) if place == "left" else ((-5387 / 26584376)*(x - area_place[0][1])**3 + (501259 / 26584376)*(x - area_place[0][1])**2 + (-81073 / 255619)*(x - area_place[0][1]) + 6)
                already_angle = math.atan(z / x)
                result_angle = already_angle + (2*math.pi * (degree/360))
                new_x = round(math.cos(result_angle) * sqrt(z ** 2 + x ** 2))
                degree_outer_eye_picture[y][new_x] = outer_eye_picture[y][x]
                if degree_outer_eye_picture[y][new_x-2] == degree_outer_eye_picture[y][new_x] == 10:
                    degree_outer_eye_picture[y][new_x-1] = 10
        return degree_outer_eye_picture
    return outer_eye_picture


def mouth_area(degree):
    """B=(31 / 58320000) * x ** 4 - (31 / 162000) * x ** 3 + (1499 / 64800) * x ** 2 - (383 / 360) * x
       -B=(-1/270) * x ** 2 + (2/3) * x
       h = ((-1/90)*x**2 + 2*x)
       """
    area_place = raw_points_0_degree["mouth_point_array"]
    mouth_picture = np.zeros(BLACK_PICTURE_0_DEGREE.shape)
    for y in range(-15, 30):
        for x in range(0, area_place[2][1] - area_place[0][1]):
            mouth_picture[area_place[0][0] + y][area_place[0][1] + x] = 10 if ((31 / 58320000) * x ** 4 - (31 / 162000) * x ** 3 + (1499 / 64800) * x ** 2 - (383 / 360) * x) <= y <= ((-1 / 270) * x ** 2 + (2 / 3) * x) else 0
    if not degree == 0:
        degree_mouth_picture = np.zeros(BLACK_PICTURE_0_DEGREE.shape)
        for y in range(area_place[0][0] - 15, area_place[0][0] + 30):
            for x in range(area_place[0][1], area_place[2][1]):
                z = ((-1 / 90) * (x - area_place[0][1]) ** 2 + 2 * (x - area_place[0][1]))
                already_angle = math.atan(z / x)
                result_angle = already_angle + (2*math.pi * (degree/360))
                new_x = round((math.cos(result_angle) * sqrt(z ** 2 + x ** 2)))
                degree_mouth_picture[y][new_x] = mouth_picture[y][x]
                if degree_mouth_picture[y][new_x-2] == degree_mouth_picture[y][new_x] == 10:
                    degree_mouth_picture[y][new_x-1] = 10
        return degree_mouth_picture
    return mouth_picture


def save_raw_face_net():
    json = dumps(raw_face_net_0_degree)
    file = open("/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/raw_face_net_dict.json", "w")
    file.write(json)
    file.close()


def save_raw_points():
    json = dumps(raw_points_0_degree)
    file = open("/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/points_0_degree_dict.json", "w")
    file.write(json)
    file.close()


def save_all_face_nets(face_net_dict, path):
    json = dumps(face_net_dict)
    file = open(path, 'w')
    file.write(json)
    file.close()


def save_face_net(face_net, degree):
    json = dumps(face_net)
    file = open("/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/face_net_"+str(degree)+".json", "w")
    file.write(json)
    file.close()


if __name__ == "__main__":
    """
    for i in range(-20, 20):
        face_net_picture = draw_net(calculate_net(i))
        # io.imsave("/home/bernihoh/Bachelor/SMS/modification/netcompare/picture/face_net_"+str(i)+"degree.png", face_net_picture)
    
    face_net_dict = {}
    for i in range(-20, 21):
        face_net = calculate_net(i)
        for area in face_net["areas"]:
            list_area = face_net["areas"][area].tolist()
            face_net["areas"][area] = list_area
        face_net_dict.update({i: face_net})
    save_all_face_nets(face_net_dict)
    """

    for i in range(-20, 21):
        face_net = calculate_net(i)
        for area in face_net["areas"]:
            list_area = face_net["areas"][area].tolist()
            face_net["areas"][area] = list_area
        save_face_net(face_net, i)
    #save_raw_face_net()
    #save_raw_points()

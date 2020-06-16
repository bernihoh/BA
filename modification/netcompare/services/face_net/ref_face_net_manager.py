from json import dumps, load
import pprint
from typing import List, Any, Union
from skimage import io
from skimage.draw import line
from skimage import io
import os
import sys
import math
from math import *
import numpy as np
import pickle as pkl
import matplotlib
import matplotlib.pyplot as plt
import codecs
# from MaskRCNN.samples.SMSNetworks.SMSNetworks import FFN
from modification.helper_files import roi_helper
from modification.helper_files import math_helper
from multiprocessing import *


def ref_face_net_0_degree():
    rfn0d_point_h = \
        {"iris_center_point_l": -30,
         "iris_center_point_r": -30,
         "inner_eye_point_array_l": [-75, -35],
         "inner_eye_point_array_r": [-35, -75],
         "outer_eye_point_array_l": [-105, -25],
         "outer_eye_point_array_r": [-25, -105],
         "cheek_point_array_l": [-135, -90],
         "cheek_point_array_r": [-90, -135],
         "eye_brow_point_array_l": [-90, -60, -30, -15, 0],
         "eye_brow_point_array_r": [-0, -15, -30, -60, -90],
         "eye_bridge_point": 0,
         "nose_tip_point_array": [-5, 90, -5],
         "mouth_point_array": [-30, 35, -30],
         "chin_point_array": [-30, -5, -30]
         }
    rfn0d = \
        {"points": {
            "iris_center_point_l": [90, 165],
            "iris_center_point_r": [90, 415],
            "inner_eye_point_array_l": [[90, 110], [90, 210]],
            "inner_eye_point_array_r": [[90, 370], [90, 470]],
            "outer_eye_point_array_l": [[90, 80], [90, 230]],
            "outer_eye_point_array_r": [[90, 350], [90, 500]],
            "cheek_point_array_l": [[180, 60], [240, 115]],
            "cheek_point_array_r": [[240, 465], [180, 520]],
            "eye_brow_point_array_l": [[45, 80], [20, 110], [15, 165], [30, 210], [45, 230]],
            "eye_brow_point_array_r": [[45, 350], [30, 370], [15, 415], [20, 470], [45, 500]],
            "eye_bridge_point": [90, 290],
            "nose_tip_point_array": [[225, 210], [210, 290], [225, 370]],
            "mouth_point_array": [[345, 170], [360, 290], [345, 410]],
            "chin_point_array": [[450, 230], [420, 290], [450, 350]],
        },
            "areas": {
                "iris_area_l": calculate_ref_iris_area(),
                "iris_area_r": calculate_ref_iris_area(),
                "inner_eye_area_l": calculate_ref_inner_eye_area(),
                "inner_eye_area_r": calculate_ref_inner_eye_area(),
                "outer_eye_area_l": calculate_ref_outer_eye_area(),
                "outer_eye_area_r": calculate_ref_outer_eye_area(),
                "mouth_area": calculate_ref_mouth_area(),
            }
        }
    return [rfn0d, rfn0d_point_h]


def rotate_rfn0d(degree):
    rot_c = [290, 0]  # this point is in the x, z plane and No y
    rfn0d, rfn0d_ph = ref_face_net_0_degree()
    anchor_p = rfn0d["points"]["eye_bridge_point"]
    rfn_d = {
        "points": {
            "iris_center_point_l": [rfn0d["points"]["iris_center_point_l"][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["iris_center_point_l"][1], rfn0d_ph["iris_center_point_l"]]), degree)[0]],
            "iris_center_point_r": [rfn0d["points"]["iris_center_point_r"][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["iris_center_point_r"][1], rfn0d_ph["iris_center_point_r"]]), degree)[0]],
            "inner_eye_point_array_l": [[rfn0d["points"]["inner_eye_point_array_l"][0][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["inner_eye_point_array_l"][0][1], rfn0d_ph["inner_eye_point_array_l"][0]]), degree)[0]],
                                        [rfn0d["points"]["inner_eye_point_array_l"][-1][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["inner_eye_point_array_l"][-1][1], rfn0d_ph["inner_eye_point_array_l"][-1]]), degree)[0]]],
            "inner_eye_point_array_r": [[rfn0d["points"]["inner_eye_point_array_r"][0][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["inner_eye_point_array_r"][0][1], rfn0d_ph["inner_eye_point_array_r"][0]]), degree)[0]],
                                        [rfn0d["points"]["inner_eye_point_array_r"][-1][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["inner_eye_point_array_r"][-1][1], rfn0d_ph["inner_eye_point_array_r"][-1]]), degree)[0]]],
            "outer_eye_point_array_l": [[rfn0d["points"]["outer_eye_point_array_l"][0][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["outer_eye_point_array_l"][0][1], rfn0d_ph["outer_eye_point_array_l"][0]]), degree)[0]],
                                        [rfn0d["points"]["outer_eye_point_array_l"][-1][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["outer_eye_point_array_l"][-1][1], rfn0d_ph["outer_eye_point_array_l"][-1]]), degree)[0]]],
            "outer_eye_point_array_r": [[rfn0d["points"]["outer_eye_point_array_r"][0][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["outer_eye_point_array_r"][0][1], rfn0d_ph["outer_eye_point_array_r"][0]]), degree)[0]],
                                        [rfn0d["points"]["outer_eye_point_array_r"][-1][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["outer_eye_point_array_r"][-1][1], rfn0d_ph["outer_eye_point_array_r"][-1]]), degree)[0]]],
            "cheek_point_array_l": [[rfn0d["points"]["cheek_point_array_l"][0][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["cheek_point_array_l"][0][1], rfn0d_ph["cheek_point_array_l"][0]]), degree)[0]],
                                    [rfn0d["points"]["cheek_point_array_l"][-1][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["cheek_point_array_l"][-1][1], rfn0d_ph["cheek_point_array_l"][-1]]), degree)[0]]],
            "cheek_point_array_r": [[rfn0d["points"]["cheek_point_array_r"][0][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["cheek_point_array_r"][0][1], rfn0d_ph["cheek_point_array_r"][0]]), degree)[0]],
                                    [rfn0d["points"]["cheek_point_array_r"][-1][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["cheek_point_array_r"][-1][1], rfn0d_ph["cheek_point_array_r"][-1]]), degree)[0]]],
            "eye_brow_point_array_l": [[rfn0d["points"]["eye_brow_point_array_l"][0][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["eye_brow_point_array_l"][0][1], rfn0d_ph["eye_brow_point_array_l"][0]]), degree)[0]],
                                       [rfn0d["points"]["eye_brow_point_array_l"][1][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["eye_brow_point_array_l"][1][1], rfn0d_ph["eye_brow_point_array_l"][1]]), degree)[0]],
                                       [rfn0d["points"]["eye_brow_point_array_l"][2][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["eye_brow_point_array_l"][2][1], rfn0d_ph["eye_brow_point_array_l"][2]]), degree)[0]],
                                       [rfn0d["points"]["eye_brow_point_array_l"][3][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["eye_brow_point_array_l"][3][1], rfn0d_ph["eye_brow_point_array_l"][3]]), degree)[0]],
                                       [rfn0d["points"]["eye_brow_point_array_l"][-1][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["eye_brow_point_array_l"][-1][1], rfn0d_ph["eye_brow_point_array_l"][-1]]), degree)[0]]],
            "eye_brow_point_array_r": [[rfn0d["points"]["eye_brow_point_array_r"][0][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["eye_brow_point_array_r"][0][1], rfn0d_ph["eye_brow_point_array_r"][0]]), degree)[0]],
                                       [rfn0d["points"]["eye_brow_point_array_r"][1][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["eye_brow_point_array_r"][1][1], rfn0d_ph["eye_brow_point_array_r"][1]]), degree)[0]],
                                       [rfn0d["points"]["eye_brow_point_array_r"][2][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["eye_brow_point_array_r"][2][1], rfn0d_ph["eye_brow_point_array_r"][2]]), degree)[0]],
                                       [rfn0d["points"]["eye_brow_point_array_r"][3][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["eye_brow_point_array_r"][3][1], rfn0d_ph["eye_brow_point_array_r"][3]]), degree)[0]],
                                       [rfn0d["points"]["eye_brow_point_array_r"][-1][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["eye_brow_point_array_r"][-1][1], rfn0d_ph["eye_brow_point_array_r"][-1]]), degree)[0]]],
            "eye_bridge_point": [90, 290],
            "nose_tip_point_array": [[rfn0d["points"]["nose_tip_point_array"][0][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["nose_tip_point_array"][0][1], rfn0d_ph["nose_tip_point_array"][0]]), degree)[0]],
                                     [rfn0d["points"]["nose_tip_point_array"][1][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["nose_tip_point_array"][1][1], rfn0d_ph["nose_tip_point_array"][1]]), degree)[0]],
                                     [rfn0d["points"]["nose_tip_point_array"][-1][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["nose_tip_point_array"][-1][1], rfn0d_ph["nose_tip_point_array"][-1]]), degree)[0]]],
            "mouth_point_array": [[rfn0d["points"]["mouth_point_array"][0][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["mouth_point_array"][0][1], rfn0d_ph["mouth_point_array"][0]]), degree)[0]],
                                     [rfn0d["points"]["mouth_point_array"][1][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["mouth_point_array"][1][1], rfn0d_ph["mouth_point_array"][1]]), degree)[0]],
                                     [rfn0d["points"]["mouth_point_array"][-1][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["mouth_point_array"][-1][1], rfn0d_ph["mouth_point_array"][-1]]), degree)[0]]],
            "chin_point_array": [[rfn0d["points"]["chin_point_array"][0][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["chin_point_array"][0][1], rfn0d_ph["chin_point_array"][0]]), degree)[0]],
                                  [rfn0d["points"]["chin_point_array"][1][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["chin_point_array"][1][1], rfn0d_ph["chin_point_array"][1]]), degree)[0]],
                                  [rfn0d["points"]["chin_point_array"][-1][0], anchor_p[1] + math_helper.calculate_point_rot(math_helper.calculate_vector(rot_c, [rfn0d["points"]["chin_point_array"][-1][1], rfn0d_ph["chin_point_array"][-1]]), degree)[0]]],
        },
        "areas": {
            "iris_area_l": calculate_iris_rot(rfn0d["areas"]["iris_area_l"], degree),
            "iris_area_r": calculate_iris_rot(rfn0d["areas"]["iris_area_r"], degree),
            "inner_eye_area_l": calculate_inner_eye_l_rot(rfn0d["areas"]["inner_eye_area_l"], degree),
            "inner_eye_area_r": calculate_inner_eye_r_rot(rfn0d["areas"]["inner_eye_area_r"], degree),
            "outer_eye_area_l": calculate_outer_eye_l_rot(rfn0d["areas"]["outer_eye_area_l"], degree),
            "outer_eye_area_r": calculate_outer_eye_r_rot(rfn0d["areas"]["outer_eye_area_r"], degree),
            "mouth_area": calculate_mouth_rot(rfn0d["areas"]["mouth_area"], degree)
        }
    }

    return rfn_d


def calculate_ref_iris_area():
    """b_u, b_d: Everything within 15px of iris_center_point"""
    img = np.zeros((30, 45))
    icp = [15, 23]
    for y in range(icp[0] - 15, icp[0] + 15):
        for x in range(icp[1] - 15, icp[1] + 15):
            img[y, x] = 1 if 15 >= sqrt((y - icp[0]) ** 2 + (x - icp[1]) ** 2) else 0
    return img


def calculate_ref_inner_eye_area():
    b_u = lambda x: ((-15)/2401)*x**2 + (30/49)*x + 15
    b_d = lambda x: (15/2401)*x**2 + ((-30)/49)*x + 15
    img = calculate_area(b_u, b_d, (30, 100))
    return img


def calculate_ref_outer_eye_area():
    b_u = lambda x: (2/375)*x**2 + ((-4)/5)*x + 45
    b_d = lambda x: ((-1)/125)*x**2 + (6/5)*x + 45
    img = calculate_area(b_d, b_u, (100, 150))
    return img


def calculate_ref_mouth_area():
    b_u = lambda x: (1/480)*x**2 + (-1/2)*x + 30
    b_d = lambda x: ((-1)/240)*x**2 + x + 30
    img = calculate_area(b_d, b_u, (90, 240))
    return img


def calculate_area(f_u, f_d, img_shape):
    img = np.zeros(img_shape)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            img[img.shape[0] - 1 - y, x] = 1 if f_d(x) < img.shape[0] - 1 - y < f_u(x) else 0
    return img


def calculate_iris_rot(iris_area, degree):
    h = lambda x: (-1/45)*x**2 + (2/3)*x - 35
    return math_helper.calculate_area_rot(iris_area, h, degree)


def calculate_inner_eye_l_rot(innere_eye_area, degree):
    h = lambda x: (91675/536594688)*x**3 + (-9520555/268297344)*x**2 + (24605995/10950912)*x - 75
    return math_helper.calculate_area_rot(innere_eye_area, h, degree)


def calculate_inner_eye_r_rot(innere_eye_area, degree):
    h = lambda x: (-10725/59621632)*x**3 + (463445/29810816)*x**2 + (-272205/1216768)*x - 35
    return math_helper.calculate_area_rot(innere_eye_area, h, degree)


def calculate_outer_eye_l_rot(outre_eye_area, degree):
    h = lambda x: (8/84375)*x**3 + (-11/375)*x**2 + (14/5)*x - 105
    return math_helper.calculate_area_rot(outre_eye_area, h, degree)


def calculate_outer_eye_r_rot(outre_eye_area, degree):
    h = lambda x: (-8/84375)*x**3 + (1/75)*x**2 + (-2/5*x) - 25
    return math_helper.calculate_area_rot(outre_eye_area, h, degree)


def calculate_mouth_rot(mouth_area, degree):
    h = lambda x: (-13/2880)*x**2 + (13/12)*x - 30
    return math_helper.calculate_area_rot(mouth_area, h, degree)


"""
def calculate_point_rot(vector, degree):
    # rotates only in the x, z plane
    rad_a = radians(degree)
    rot_m = np.asarray([[cos(rad_a), -sin(rad_a)],
                        [sin(rad_a), cos(rad_a)]])
    n_vector = np.multiply(rot_m, vector)
    return n_vector
"""


def parallel_rotate_rfn0d(degree_list, path):
    return_list = []
    for degree in degree_list:
        return_list.append(rotate_rfn0d(degree))
        # return_list.append([0])
    for rfn_dict, degree in zip(return_list, degree_list):
        file = open(path+"face_net_pkl_ref_"+str(degree)+"_degree.pkl", 'wb')
        pkl.dump(rfn_dict, file)
        file.close()
    return


if __name__ == "__main__":
    """

    path = "/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/face_net_pkl_ref/"
    file = open(path + "face_net_pkl_ref_0_degree.pkl", 'wb')
    rfn0d = ref_face_net_0_degree()[0]
    pkl.dump(rfn0d, file)
    file.close()
    """

    """
    path = "/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/face_net_pkl_ref/"
    rfn0d = ref_face_net_0_degree()[0]
    print(rfn0d)
    """

    path = "/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/face_net_pkl_ref/"
    pool = Pool(cpu_count())
    degree_list = list(range(-30, 31))
    #print(degree_list)
    chunk_length = int(np.round(len(degree_list) / cpu_count()))
    for i in range(cpu_count()):
        degree_list_chunk = degree_list[i * chunk_length: (i + 1) * chunk_length] if not i == cpu_count() - 1 else degree_list[i * chunk_length: len(degree_list)]
        pool.apply_async(parallel_rotate_rfn0d, args=(degree_list_chunk, path))
    pool.close()

    """
    path = "/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/face_net_pkl_ref/"
    for i in range(-30, 31):
        print("Progress", str(i+30) + "/" + str(60))
        rot_rfn = rotate_rfn0d(i)
        file = open(path + "face_net_pkl_ref_"+str(i)+"_degree.pkl", 'wb')
        pkl.dump(rot_rfn, file)
        file.close()
    """
    """
    path = "/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/face_net_pkl_ref/"
    pickle_files = os.listdir(path)
    for file, i in zip(pickle_files, range(len(pickle_files))):
        rfn_dict = pkl.load(open(path+file, 'rb'))
        picture = rfn_dict["areas"]["iris_area_l"]
        plt.imshow(picture)
        plt.title(i)
        plt.show()
        print(i, rfn_dict["areas"]["mouth_area"])
    """

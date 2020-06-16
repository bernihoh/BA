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
from MaskRCNN.samples.SMSNetworks.SMSNetworks import FFN
from modification.helper_files import roi_helper
from modification.helper_files import math_helper
from multiprocessing import *
import random
from typing import Any
from math import pi, sin, cos
from skimage.draw import polygon
from skimage import io
import cv2
import math
import modification.icp.icp as icp
import MaskRCNN.samples.SMSNetworks.SMSNetworks as SMSNetworks
from stylegan2.run_generator import generate_imgs as generate_heads
import modification.netcompare.netcompare as netcompare
from modification.pic_fitting_tool.pic_fitting import pic_fitting, edge_detection


def transpose(matrix):
    plt.imshow(matrix)
    plt.show()
    m_t = matrix.T
    plt.imshow(m_t)
    plt.show()
    return m_t


def array_harmonizing(va_0,  va_1, l_s, r_s, arr):

    """calculates the harmonizing color flow of the array 'arr' between the pt-value 'va_0' left of 'arr' and pt-value 'va_1' right of 'arr'."""

    ls_pt = len(arr) * l_s
    rs_pt = len(arr) * r_s
    hrzsc_l = (pi/2) / (l_s * len(arr)) if l_s > 0 else 0
    hrzsc_r = (pi/2) / (len(arr) - r_s*len(arr)) if r_s < 1 else 0
    sin2_l = lambda x: sin(hrzsc_l*x)**2
    sin2_r = lambda x: sin(hrzsc_r*x)**2
    cos2_l = lambda x: cos(hrzsc_l*x)**2
    cos2_r = lambda x: cos(hrzsc_r*x)**2
    h_arr = np.zeros(arr.shape)
    if len(arr.shape) > 2:
        for i in range(arr.shape[0]):
            if i <= ls_pt:
                h_arr[i] = np.asarray([va_0i*cos2_l(i) + v_arri*sin2_l(i) for va_0i, v_arri in zip(va_0, arr[i])])
            elif ls_pt < i <= rs_pt:
                h_arr[i] = arr[i]
            else:
                h_arr[i] = np.asarray([va_1i*sin2_r(i-rs_pt) + v_arri*cos2_r(i-rs_pt) for va_1i, v_arri in zip(va_1, arr[i])])

    else:
        for i in range(arr.shape[0]):
            if i <= ls_pt:
                h_arr[i] = va_0*cos2_l(i) + arr[i]*sin2_l(i)
            elif ls_pt < i <= rs_pt:
                h_arr[i] = arr[i]
            else:
                h_arr[i] = va_1*sin2_r(i-rs_pt) + arr[i]*cos2_r(i-rs_pt)

    #h_arr = arr
    return h_arr


def gan_train_input(height, width, amount, save_path):
    #_zero = np.zeros((height, width), dtype=np.float)
    _ones = np.full((height, width), 1, dtype=np.float)
    rel_split_mean = [15, 85]
    _mix = np.zeros((height, width), dtype=np.float)
    for a in range(amount[0], amount[1]):
        diff = np.random.randint(3, 12)
        for y in range(height):
            rel_split = [np.random.randint(rel_split_mean[0] - diff, rel_split_mean[0] + diff)/100,
                         np.random.randint(rel_split_mean[1] - diff, rel_split_mean[1] + diff)/100]
            _mix[y, :] = array_harmonizing(0, 0, rel_split[0], rel_split[1], _ones[y, :])
        #plt.imshow(_mix)
        #plt.show()
        diff = np.random.randint(3, 12)
        for x in range(width):
            rel_split = [np.random.randint(rel_split_mean[0] - diff, rel_split_mean[0] + diff) / 100,
                         np.random.randint(rel_split_mean[1] - diff, rel_split_mean[1] + diff) / 100]
            _mix[:, x] = array_harmonizing(0, 0, rel_split[0], rel_split[1], _mix[:, x])

        plt.imshow(_mix, cmap='gray')
        plt.show()
        #io.imsave(save_path+str(a)+".png", _mix)
        _mix_c = np.subtract(_ones, _mix)
        #plt.imshow(_mix_c)
        #plt.show()


if __name__ == "__main__":
    save_path = "/home/bernihoh/Bachelor/SMS/modification/wallmod/mix_pic_train/"
    gan_train_input(100, 100, [0, 100], save_path)
    """
    path = "/home/bernihoh/Bachelor/SMS/modification/wallmod/mix_pic_train/"
    file_list = os.listdir(path)
    matrix_array = [io.imread(path+file) for file in file_list]
    for matrix, i in zip(matrix_array, range(len(matrix_array))):
        m_t = transpose(matrix)
        io.imsave(path+str(i)+"T.png", m_t)

    """




















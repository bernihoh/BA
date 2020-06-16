import random
from typing import Any
import numpy as np
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
import modification.helper_files.roi_helper as roi_helper
import modification.helper_files.math_helper as math_helper
import matplotlib.pyplot as plt
from MaskRCNN.samples.SMSNetworks import SMSNetworks
import os
import skimage.io as skio
import tensorflow as tf
from IPython import display
import pandas as pd
import pickle as pkl
import glob
import imageio
import os
import PIL
from tensorflow.keras import layers
import time


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(25*25*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((25, 25, 256)))
    assert model.output_shape == (None, 25, 25, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 25, 25, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 50, 50, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 100, 100, 1)

    return model


class decorate_walls:

    def __init__(self, spade_pic, label_pic):
        self.spd_pic = cv2.resize(spade_pic, (256, 256))
        self.lbl_pic = cv2.resize(label_pic, (256, 256))
        return

    def decorate_walls(self):

        txtr_path = "/home/bernihoh/Bachelor/SMS/modification/wallmod/texture_pics/"
        wfn_res = SMSNetworks.WFN(self.lbl_pic).detect()

        generator = make_generator_model()
        generator.load_weights("/home/bernihoh/Bachelor/SMS/modification/wallmod/logs/cp.ckpt/generator_ckpt/")
        mix_pics = generator(tf.random.normal([len(wfn_res["rois"]), 100]), training=False)  # tf.random.normal([len(wfn_res["rois"])
        # print(np.asarray(mix_pics[1, :, :, 0].eval().tolist()))
        # plt.imshow(np.asarray(mix_pics[1, :, :, 0].eval().tolist()), cmap = 'gray')
        # plt.show()
        for i, roi, class_name in zip(range(len(wfn_res["rois"])), wfn_res["rois"], wfn_res["class_names"]):
            #if not (self.lbl_pic[roi[0], roi[1]][0] == 192 and self.lbl_pic[roi[0], roi[1]][1] == 160 and self.lbl_pic[roi[0], roi[1]][2] == 64) and \
            #   not (self.lbl_pic[roi[0], roi[1]][0] == 64 and self.lbl_pic[roi[0], roi[1]][1] == 160 and self.lbl_pic[roi[0], roi[1]][2] == 64):
             #   continue
            txtr = io.imread(txtr_path+class_name[0]+"_"+class_name[1]+".png")
            plt.imshow(txtr)
            plt.show()
            txtr = cv2.resize(txtr, (roi[3] - roi[1], roi[2] - roi[0]), interpolation=cv2.INTER_AREA)
            mix_pic = np.asarray(mix_pics[i, :, :, 0].eval().tolist())
            mp_min, mp_max = np.amin(mix_pic), np.amax(mix_pic)
            mix_pic = (mix_pic + np.full(mix_pic.shape, 0 - mp_min)) / mp_max
            #print(mix_pic)
            plt.imshow(mix_pic, cmap='gray')
            plt.title("mix_pic 100,100")
            plt.show()
            mix_pic = cv2.resize(mix_pic, (roi[3] - roi[1], roi[2] - roi[0]), interpolation=cv2.INTER_AREA)
            plt.imshow(mix_pic, cmap='gray')
            plt.title("mix_pic resized")
            plt.show()
            ground = self.spd_pic[roi[0]: roi[2], roi[1]: roi[3]]
            # print("######################################################################## ", self.lbl_pic[roi[0], roi[1]])
            plt.imshow(ground)
            plt.title("ground")
            plt.show()
            rev_mp = np.subtract(np.full((roi[2] - roi[0], roi[3] - roi[1]), 1.0), mix_pic).astype(np.float)
            plt.imshow(rev_mp)
            plt.title("rev_mp")
            plt.show()
            spd_prt = np.zeros(ground.shape)
            for i in range(ground.shape[2]):
                spd_prt[:, :, i] = ground[:, :, i] * rev_mp
            spd_prt = spd_prt.astype(dtype=np.uint8)
            plt.imshow(spd_prt)
            plt.title("spd_prt")
            plt.show()
            txtr_prt = np.zeros(txtr.shape)
            for i in range(ground.shape[2]):
                txtr_prt[:, :, i] = txtr[:, :, i] * mix_pic
            txtr_prt = txtr_prt.astype(dtype=np.uint8)
            plt.imshow(txtr_prt)
            plt.title("txtr_prt")
            plt.show()
            self.spd_pic[roi[0]: roi[2], roi[1]: roi[3]] = spd_prt + txtr_prt
        return self.spd_pic


if __name__ == "__main__":
    spade_pic = io.imread("/home/bernihoh/Bachelor/SMS/modification/wallmod/spade_test_pic.png")
    label_pic = io.imread("/home/bernihoh/Bachelor/SMS/modification/wallmod/label_test_pic.png")
    dec_wls = decorate_walls(spade_pic, label_pic)
    plt.imshow(dec_wls.decorate_walls())
    plt.show()

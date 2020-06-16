import os
import numpy
from skimage import io
import matplotlib.pyplot as plt

picturepath = "/home/bernihoh/Bachelor/SMS/MaskRCNN/samples/SMSNetworks/wall_feature_place_detection/pic_train/"
new_picture_path = "/home/bernihoh/Bachelor/SMS/MaskRCNN/samples/SMSNetworks/wall_feature_place_detection/pic_train1/"
pictures = [io.imread(picturepath + pic_file) for pic_file in os.listdir(picturepath)]
"""
for pic, a in zip(pictures, range(0, len(pictures))):
    flipped_pic = numpy.flip(pic, 0)
    if flipped_pic.shape[2] == 4:
        flipped_pic = flipped_pic[:, :, 0:3]
    io.imsave(new_picture_path + str(a) + ".png", flipped_pic)
"""
"""    
for pic_file in os.listdir(picturepath):
    pic = io.imread(picturepath + pic_file)
    if pic.shape[2] == 4:
        pic = pic[:, :, 0:3]
        #io.imsave(new_picture_path + pic_file, pic)
"""
for pic, a in zip(pictures, range(0, len(pictures))):
    rot_90 = numpy.rot90(pic, axes=(0, 1))
    rot_180 = numpy.rot90(rot_90, axes=(0, 1))
    rot_270 = numpy.rot90(rot_180, axes=(0, 1))
    num_arr = numpy.arange(a*4, (a+1)*4, dtype=int)
    #io.imsave(new_picture_path + str(num_arr[0]) + ".png", pic)
    #io.imsave(new_picture_path + str(num_arr[1]) + ".png", rot_90)
    #io.imsave(new_picture_path + str(num_arr[2]) + ".png", rot_180)
    #io.imsave(new_picture_path + str(num_arr[3]) + ".png", rot_270)

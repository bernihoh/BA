import os
import numpy
from skimage import io, data, color
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
import pickle as pkl
"""
path_load = "/home/bernihoh/Bachelor/SPADE/results/coco_pretrained/test_latest/images/input_label/"
path_save = "/home/bernihoh/Bachelor/SPADE/results/coco_pretrained/test_latest/images/input_label1/"
files = os.listdir(path_load)
for file_name, i in zip(files, range(len(files))):
    file = open(path_load+file_name, 'rb')
    pkl_file = pkl.load(file)
    save_file_path = open(path_save + str(i) + ".png", 'wb')
    pkl.dump(pkl_file, save_file_path)
    save_file_path.close()
"""
path_load = "/home/bernihoh/Bachelor/SMS/MaskRCNN/samples/SMSNetworks/wall_feature_place_detection/pic_val/"
path_save = "/home/bernihoh/Bachelor/SMS/MaskRCNN/samples/SMSNetworks/wall_feature_place_detection/pic_val/"
pictures = [io.imread(path_load + pic_file) for pic_file in os.listdir(path_load)]

#for pic, a in zip(pictures, range(0, len(pictures))):
    #io.imsave(path_save + str(a) + ".png", pic)

#pic_files = os.listdir(path_load)
#for file in pic_files:
#    if file[-5] == "T":
#        os.remove(path_load+file)

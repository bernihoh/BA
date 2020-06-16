import os
import numpy
from skimage import io, data, color
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
import pickle as pkl

picturepath = "/home/bernihoh/Bachelor/SMS/SPADE/datasets/coco_stuff/example_val_img/"
pic_files = os.listdir(picturepath)
for pic_file, a in zip(pic_files, range(0, len(pic_files))):
    print(a, len(pic_files))
    pic = io.imread(picturepath + pic_file)
    #if pic.shape[0] == 1024 or pic.shape[1] == 1024:
    pic = resize(pic, (256, 256), anti_aliasing=True)
    #io.imshow(pic)
    #plt.show()
    io.imsave("/home/bernihoh/Bachelor/abschlussarbeiten/abschlussarbeit pics/" + str(a) + "_label.png", pic)


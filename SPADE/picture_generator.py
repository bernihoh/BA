import os
from collections import OrderedDict
import matplotlib.pyplot as plt
import SPADE.data as data
from SPADE.options.test_options import TestOptions
from SPADE.models.pix2pix_model import Pix2PixModel
from SPADE.util.visualizer import Visualizer
from SPADE.util import html
from numba import cuda
import numpy as np
import tensorflow as tf
from importlib import import_module
import sys
from PIL import Image


class pic_gen:

    def generate():
        cuda.select_device(0)
        opt = TestOptions().parse()
        dataloader = data.create_dataloader(opt)
        model = Pix2PixModel(opt)
        model.eval()
        visualizer = Visualizer(opt)
        # create a webpage that summarizes the all results
        web_dir = os.path.join(opt.results_dir, opt.name,
                               '%s_%s' % (opt.phase, opt.which_epoch))
        webpage = html.HTML(web_dir,
                            'Experiment = %s, Phase = %s, Epoch = %s' %
                            (opt.name, opt.phase, opt.which_epoch))
        # test
        for i, data_i in enumerate(dataloader):
            if i * opt.batchSize >= opt.how_many:
                break

            generated = model(data_i, mode='inference')
            print(i)
            img_path = data_i['path']
            web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
            webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
            for b in range(generated.shape[0]):
                print(b)
                print('process image... %s' % img_path[b])
                print(data_i['label'][b])
                #visuals = OrderedDict([('input_label', data_i['label'][b]),
                 #                      ('synthesized_image', generated[b])])

                #visualizer.save_images(webpage, visuals, img_path[b:b + 1])
                #visuals = visualizer.convert_visuals_to_numpy(visuals)
        #cuda.close()
        #return visuals
        return 0

    def generate_1():
        cuda.select_device(0)
        opt = TestOptions().parse()
        dataloader = data.create_dataloader(opt)
        model = Pix2PixModel(opt)
        model.eval()
        visualizer = Visualizer(opt)
        web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
        return_arr = []
        for i, data_i in enumerate(dataloader):
            if i == len(dataloader)-1:
                #print(data_i)
                if i * opt.batchSize >= opt.how_many:
                    break
                generated = model(data_i, mode='inference')
                #print(generated)
                img_path = data_i['path']
                visuals = OrderedDict([('input_label', data_i['label'][0]),
                                       ('synthesized_image', generated[0])])
                #visualizer.save_images(webpage, visuals, img_path[0:0 + 1])
                for label, image_numpy in Visualizer.convert_visuals_to_numpy(visualizer, visuals).items():
                    image_numpy = np.expand_dims(image_numpy, axis=2)
                    image_numpy = np.repeat(image_numpy, 3, 2)
                    #+image_pil = Image.fromarray(image_numpy)
                    #plt.imshow(image_numpy)
                    #print(image_numpy)
                    image_numpy = image_numpy[:, :, 0, :]
                    return_arr.append(image_numpy)
            #for img in return_arr:
               # plt.imshow(img)
               # plt.show()
        return return_arr

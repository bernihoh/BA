import os
from abc import *
import json
from math import *
from stylegan2.run_generator import generate_imgs
import numpy as np
import modification.helper_files.math_helper as math_helper
import modification.helper_files.roi_helper as roi_helper
import modification.netcompare.services.face_net.face_net_manager as fnm
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle as pkl
FAULT_COUNTER_MAXIMUM = 30


class net_compare(ABC):
    def __init__(self, service):
        assert service in ["face_net"]
        self.service_path = os.path.join(os.path.dirname(__file__), "services/" + service)
        self.reference_net_path = "/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/face_net_pkl_ref/"

    # abstract method
    def compare_net(self, net):
        return "You are using the abstract method"


class face_net_compare(net_compare):
    def compare_net(self, face_net):
        if face_net == "redo":
            bs = True
            return bs
        fsi = fnm.face_score_input(face_net)[0]
        inp_arr = np.zeros((1, 243, 41))
        inp_arr[0] = fsi
        plt.imshow(fsi)
        plt.show()
        # build the model
        model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(243, 41)),
                                            tf.keras.layers.Dense(512, activation='hard_sigmoid'),
                                            tf.keras.layers.Dense(128, activation='relu'),
                                            tf.keras.layers.Dense(64, activation='relu'),
                                            tf.keras.layers.Dropout(0.382),
                                            tf.keras.layers.Dense(2)])
        # define the loss function
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # compile the model
        model.compile(optimizer='adam',
                      loss=loss_fn,
                      metrics=['accuracy'])
        load_checkpoint_path = "/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/logs/best/cp.ckpt"

        model.load_weights(load_checkpoint_path)
        predictions = model.predict_classes(inp_arr)
        #print(predictions[0])
        bs = True if predictions[0] == 1 else False
        return bs









import os
import time
# import tkinter as tk
import tkinter
from tkinter import *
from tkinter import ttk, colorchooser, filedialog, Tk
import cv2
from matplotlib import colors
import skimage
from skimage import io
# from skimage.transform import resize, rescale, downscale_local_mean
from PIL import ImageTk, Image
import io
import numpy as np
from SPADE.util import coco
from SPADE.picture_generator import pic_gen
from MaskRCNN.samples.owndemo3 import object_detection
import datetime
from modification.headmod.head_creation import head_creation
from modification.wallmod.decorate_walls import decorate_walls
import matplotlib.pyplot as plt
import random
import tensorflow as tf


class main:
    hex_number_dict = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "a": 10,
        "A": 10,
        "b": 11,
        "B": 11,
        "c": 12,
        "C": 12,
        "d": 13,
        "D": 13,
        "e": 14,
        "E": 14,
        "f": 15,
        "F": 15
    }

    def __init__(self, master):
        self.master = master
        self.color_fg = '#74dbfe'
        self.color_bg = '#74dbfe'
        self.old_x = None
        self.old_y = None
        self.penwidth = 5
        self.drawWidgets()
        self.c.bind('<B1-Motion>', self.paint)  # drwaing the line
        self.c.bind('<ButtonRelease-1>', self.reset)
        # self.rgb_grey_id_dict = {[]}
        self.rgb_grey_id_dict = {(255, 255, 255): 105, (116, 219, 254): 105, (0, 0, 255): 94, (102, 89, 136): 67,
                                 (88, 79, 47): 95, (208, 247, 144): 101, (79, 231, 140): 102, (123, 18, 183): 111,
                                 (183, 191, 98): 112, (254, 70, 246): 143, (77, 13, 152): 145, (14, 71, 30): 150,
                                 (157, 22, 112): 157, (5, 184, 220): 159, (182, 204, 190): 163, (181, 25, 231): 165,
                                 (21, 31, 9): 170, (58, 54, 76): 171, (171, 17, 181): 172, (101, 239, 64): 173,
                                 (143, 173, 167): 174, (59, 43, 100): 175, (151, 143, 72): 176, (211, 71, 110): 179,
                                 (223, 227, 157): 180, (34, 170, 255): 156, (127, 174, 158): 110, (11, 61, 110): 124,
                                 (173, 116, 26): 125, (188, 239, 105): 126, (245, 242, 34): 129, (118, 183, 178): 134,
                                 (74, 53, 26): 135, (118, 83, 253): 147, (129, 101, 109): 149, (219, 195, 134): 153,
                                 (16, 235, 242): 154, (236, 217, 254): 158, (167, 171, 194): 161, (81, 210, 236): 177,
                                 (116, 57, 117): 181, (10, 54, 90): 71, (188, 223, 215): 72, (71, 73, 164): 73,
                                 (210, 133, 53): 75, (220, 48, 140): 74, (235, 105, 182): 76, (113, 95, 204): 83,
                                 (108, 150, 23): 84, (238, 59, 179): 86, (240, 20, 161): 88, (159, 113, 74): 87,
                                 (221, 124, 195): 90, (189, 200, 162): 92, (2, 86, 34): 107, (170, 73, 103): 108,
                                 (135, 165, 205): 132, (172, 109, 208): 140, (108, 40, 24): 61, (242, 211, 233): 62,
                                 (85, 235, 210): 65, (123, 49, 237): 69, (254, 68, 70): 70, (247, 138, 219): 85,
                                 (16, 61, 174): 97, (192, 144, 3): 98, (29, 60, 7): 99, (213, 85, 9): 100,
                                 (118, 250, 8): 108, (12, 42, 95): 113, (101, 123, 156): 114, (122, 163, 234): 115,
                                 (178, 251, 199): 116, (127, 165, 78): 117, (90, 6, 11): 122, (7, 35, 165): 130,
                                 (101, 115, 31): 155, (143, 9, 5): 160, (157, 131, 86): 164, (15, 208, 185): 33,
                                 (245, 92, 214): 34, (99, 51, 69): 35, (206, 218, 61): 36, (137, 119, 250): 37,
                                 (239, 249, 163): 38, (180, 99, 85): 39, (2, 50, 99): 40, (241, 153, 153): 41,
                                 (148, 147, 235): 42, (158, 11, 48): 144, (145, 138, 145): 43, (217, 109, 64): 45,
                                 (20, 226, 70): 46, (117, 226, 108): 47, (141, 184, 52): 48, (87, 126, 43): 49,
                                 (48, 14, 138): 50, (215, 195, 145): 66, (23, 79, 70): 78, (73, 65, 113): 77,
                                 (28, 87, 95): 79, (221, 242, 137): 80, (157, 104, 189): 81, (222, 3, 4): 52,
                                 (224, 95, 39): 54, (192, 169, 42): 53, (221, 176, 17): 51, (8, 105, 83): 55,
                                 (247, 210, 14): 57, (138, 54, 76): 58, (140, 110, 40): 59, (228, 96, 133): 60,
                                 (148, 43, 202): 120, (199, 81, 14): 121, (139, 94, 97): 152, (185, 111, 212): 169,
                                 (0, 255, 0): 168, (98, 99, 19): 14, (248, 200, 172): 15, (144, 135, 32): 16,
                                 (198, 181, 180): 18, (222, 233, 183): 19, (127, 28, 209): 20, (160, 179, 153): 21,
                                 (149, 127, 16): 22, (249, 229, 214): 23, (183, 73, 54): 24, (98, 213, 15): 63,
                                 (88, 78, 38): 93, (14, 136, 9): 96, (253, 114, 124): 118, (56, 112, 12): 123,
                                 (37, 214, 38): 128, (20, 150, 136): 141, (255, 0, 0): 0, (184, 5, 220): 25,
                                 (104, 61, 56): 26, (11, 64, 182): 27, (36, 10, 72): 28, (200, 16, 20): 31,
                                 (237, 95, 41): 103, (165, 84, 98): 104, (228, 219, 82): 109, (131, 122, 203): 136,
                                 (108, 83, 81): 142, (76, 173, 99): 162, (249, 28, 28): 166, (70, 37, 245): 167}
        self.last_synth_img = None
        self.last_SPADE_label = None
        # self.rgb_grey_id_dict = {}

    def load_label_map(self, path=None):
        last_color_fg = self.color_fg
        if not path:
            file_explorer = root
            file_explorer.option_add('*foreground', 'black')
            file_explorer.option_add('*activeForeground', 'black')
            style = ttk.Style(file_explorer)
            style.configure('TLabel', foreground='black')
            style.configure('TEntry', foreground='black')
            style.configure('TMenubutton', foreground='black')
            style.configure('TButton', foreground='black')
            path = filedialog.askopenfilename(parent=root, title="Choose a png label map...")
        try:
            label_map = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)  # , (256, 256), interpolation=)
            plt.imshow(label_map)
            plt.show()
            for y in range(label_map.shape[0]):
                for x in range(label_map.shape[1]):
                    self.color_fg = colors.to_hex(label_map[y, x] / 255)
                    self.c.create_line(x, y, x + 1, y, width=1, fill=self.color_fg, capstyle=ROUND, smooth=True)

            self.color_fg = last_color_fg
        except:
            return

    def load_example(self):
        example_path = "/home/bernihoh/Bachelor/SMS/SPADE/datasets/coco_stuff/example_val_img/"
        example_file_name = random.choice(os.listdir(example_path))
        self.load_label_map(path=example_path + example_file_name)

    def paint(self, e):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, e.x, e.y, width=self.penwidth, fill=self.color_fg,
                               capstyle=ROUND, smooth=True)

        self.old_x = e.x
        self.old_y = e.y

    def reset(self, e):  # reseting or cleaning the canvas
        self.old_x = None
        self.old_y = None

    def changeW(self, e):  # change Width of pen through slider
        self.penwidth = e

    def clear(self):
        self.c.delete(ALL)

    def rgb_grey_id_dict_manager(self, rgb_hex, string):
        """
        counter = 1
        key_member = 0
        rgb_key = [0, 0, 0]
        rgb_key_counter = 0
        for char in rgb_hex:
            if not char == "#":
                if counter == 1:
                    key_member += self.hex_number_dict[char] * 16
                    counter = 0
                elif counter == 0:
                    key_member += self.hex_number_dict[char]
                    rgb_key[rgb_key_counter] = key_member
                    rgb_key_counter += 1
                    key_member = 0
                    counter = 1
        rgb_key = tuple(rgb_key)
        if rgb_key not in self.rgb_grey_id_dict:
            for i in range(182):
                ref_string = coco.id2label(i)
                if string == ref_string:
                    self.rgb_grey_id_dict.update({rgb_key: i})
        print(rgb_key)
        txt = str(self.rgb_grey_id_dict)
        print(txt)
        """

    # def change_fg(self, class_):  # changing the pen color
    #   self.color_fg = colorchooser.askcolor(color=self.color_fg)[1]

    def tree_fg(self):  # changing the pen color
        self.color_fg = "#00ff00"
        self.rgb_grey_id_dict_manager("#00ff00", "tree")

    def person_fg(self):  # changing the pen color
        self.color_fg = "#ff0000"
        self.rgb_grey_id_dict_manager("#ff0000", "person")

    def bridge_fg(self):  # changing the pen color
        self.color_fg = "#0000ff"
        self.rgb_grey_id_dict_manager("#0000ff", "bridge")

    def sky_other_fg(self):  # changing the pen color
        self.color_fg = "#22aaff"
        self.rgb_grey_id_dict_manager("#22aaff", "sky-other")

    def bicycle_fg(self):
        self.color_fg = "#606f74"
        self.rgb_grey_id_dict_manager("#606f74", "bicycle")

    def car_fg(self):
        self.color_fg = "#180d2f"
        self.rgb_grey_id_dict_manager("#180d2f", "car")

    def motorcycle_fg(self):
        self.color_fg = "#626313"
        self.rgb_grey_id_dict_manager("#626313", "motorcycle")

    def bus_fg(self):
        self.color_fg = "#ebdbe4"
        self.rgb_grey_id_dict_manager("#ebdbe4", "bus")

    def train_fg(self):
        self.color_fg = "#ed261d"
        self.rgb_grey_id_dict_manager("#ed261d", "train")

    def truck_fg(self):
        self.color_fg = "#0e170d"
        self.rgb_grey_id_dict_manager("#0e170d", "truck")

    def boat_fg(self):
        self.color_fg = "#50426c"
        self.rgb_grey_id_dict_manager("#50426c", "boat")

    def traffic_light_fg(self):
        self.color_fg = "#e22ec5"
        self.rgb_grey_id_dict_manager("#e22ec5", "traffic light")

    def fire_hydrant_fg(self):
        self.color_fg = "#d304f5"
        self.rgb_grey_id_dict_manager("#d304f5", "fire hydrant")

    def street_sign_fg(self):
        self.color_fg = "#eaae4d"
        self.rgb_grey_id_dict_manager("#eaae4d", "street sign")

    def stop_sign_fg(self):
        self.color_fg = "#d304f5"
        self.rgb_grey_id_dict_manager("#d304f5", "stop sign")

    def parking_meter_fg(self):
        self.color_fg = "#ddd144"
        self.rgb_grey_id_dict_manager("#ddd144", "parking meter")

    def bench_fg(self):
        self.color_fg = "#626313"
        self.rgb_grey_id_dict_manager("#626313", "bench")

    def bird_fg(self):
        self.color_fg = "#f8c8ac"
        self.rgb_grey_id_dict_manager("#f8c8ac", "bird")

    def cat_fg(self):
        self.color_fg = "#908720"
        self.rgb_grey_id_dict_manager("#908720", "cat")

    def dog_fg(self):
        self.color_fg = "#e76555"
        self.rgb_grey_id_dict_manager("#e76555", "dog")

    def horse_fg(self):
        self.color_fg = "#c6b5b4"
        self.rgb_grey_id_dict_manager("#c6b5b4", "horse")

    def sheep_fg(self):
        self.color_fg = "#dee9b7"
        self.rgb_grey_id_dict_manager("#dee9b7", "sheep")

    def cow_fg(self):
        self.color_fg = "#7f1cd1"
        self.rgb_grey_id_dict_manager("#7f1cd1", "cow")

    def elephant_fg(self):
        self.color_fg = "#a0b399"
        self.rgb_grey_id_dict_manager("#a0b399", "elephant")

    def bear_fg(self):
        self.color_fg = "#957f10"
        self.rgb_grey_id_dict_manager("#957f10", "bear")

    def zebra_fg(self):
        self.color_fg = "#f9e5d6"
        self.rgb_grey_id_dict_manager("#f9e5d6", "zebra")

    def giraffe_fg(self):
        self.color_fg = "#b74936"
        self.rgb_grey_id_dict_manager("#b74936", "giraffe")

    def hat_fg(self):
        self.color_fg = "#b805dc"
        self.rgb_grey_id_dict_manager("#b805dc", "hat")

    def backpack_fg(self):
        self.color_fg = "#683d38"
        self.rgb_grey_id_dict_manager("#683d38", "backpack")

    def umbrella_fg(self):
        self.color_fg = "#0b40b6"
        self.rgb_grey_id_dict_manager("#0b40b6", "umbrella")

    def shoe_fg(self):
        self.color_fg = "#240a48"
        self.rgb_grey_id_dict_manager("#240a48", "shoe")

    def eye_glasses_fg(self):
        self.color_fg = "#a0b399"
        self.rgb_grey_id_dict_manager("#a0b399", "eye glasses")

    def handbag_fg(self):
        self.color_fg = "#b46355"
        self.rgb_grey_id_dict_manager("#b46355", "handbag")

    def tie_fg(self):
        self.color_fg = "#c81014"
        self.rgb_grey_id_dict_manager("#c81014", "tie")

    def suitcase_fg(self):
        self.color_fg = "#7f1cd1"
        self.rgb_grey_id_dict_manager("#7f1cd1", "suitcase")

    def frisbee_fg(self):
        self.color_fg = "#0fd0b9"
        self.rgb_grey_id_dict_manager("#0fd0b9", "frisbee")

    def skis_fg(self):
        self.color_fg = "#f55cd6"
        self.rgb_grey_id_dict_manager("#f55cd6", "skis")

    def snowboard_fg(self):
        self.color_fg = "#633345"
        self.rgb_grey_id_dict_manager("#633345", "snowboard")

    def sports_ball_fg(self):
        self.color_fg = "#ceda3d"
        self.rgb_grey_id_dict_manager("#ceda3d", "sports ball")

    def kite_fg(self):
        self.color_fg = "#8977fa"
        self.rgb_grey_id_dict_manager("#8977fa", "kite")

    def baseball_bat_fg(self):
        self.color_fg = "#eff9a3"
        self.rgb_grey_id_dict_manager("#eff9a3", "baseball bat")

    def baseball_glove_fg(self):
        self.color_fg = "#b46355"
        self.rgb_grey_id_dict_manager("#b46355", "baseball glove")

    def skateboard_fg(self):
        self.color_fg = "#023263"
        self.rgb_grey_id_dict_manager("#023263", "skateboard")

    def surfboard_fg(self):
        self.color_fg = "#f19999"
        self.rgb_grey_id_dict_manager("#f19999", "surfboard")

    def tennis_racket_fg(self):
        self.color_fg = "#9493eb"
        self.rgb_grey_id_dict_manager("#9493eb", "tennis racket")

    def bottle_fg(self):
        self.color_fg = "#918a91"
        self.rgb_grey_id_dict_manager("#918a91", "bottle")

    def plate_fg(self):
        self.color_fg = "#f19999"
        self.rgb_grey_id_dict_manager("#f19999", "plate")

    def wine_glass_fg(self):
        self.color_fg = "#d96d40"
        self.rgb_grey_id_dict_manager("#d96d40", "wine glass")

    def cup_fg(self):
        self.color_fg = "#14e246"
        self.rgb_grey_id_dict_manager("#14e246", "cup")

    def fork_fg(self):
        self.color_fg = "#75e26c"
        self.rgb_grey_id_dict_manager("#75e26c", "fork")

    def knife_fg(self):
        self.color_fg = "#8db834"
        self.rgb_grey_id_dict_manager("#8db834", "knife")

    def spoon_fg(self):
        self.color_fg = "#577e2b"
        self.rgb_grey_id_dict_manager("#577e2b", "spoon")

    def bowl_fg(self):
        self.color_fg = "#300e8a"
        self.rgb_grey_id_dict_manager("#300e8a", "bowl")

    def banana_fg(self):
        self.color_fg = "#ddb011"
        self.rgb_grey_id_dict_manager("#ddb011", "banana")

    def apple_fg(self):
        self.color_fg = "#de0304"
        self.rgb_grey_id_dict_manager("#de0304", "apple")

    def sandwich_fg(self):
        self.color_fg = "#c0a92a"
        self.rgb_grey_id_dict_manager("#c0a92a", "sandwich")

    def orange_fg(self):
        self.color_fg = "#e05f27"
        self.rgb_grey_id_dict_manager("#e05f27", "orange")

    def broccoli_fg(self):
        self.color_fg = "#086953"
        self.rgb_grey_id_dict_manager("#086953", "broccoli")

    def carrot_fg(self):
        self.color_fg = "#de0304"
        self.rgb_grey_id_dict_manager("#de0304", "carrot")

    def hot_dog_fg(self):
        self.color_fg = "#f7d20e"
        self.rgb_grey_id_dict_manager("#f7d20e", "hot dog")

    def pizza_fg(self):
        self.color_fg = "#8a364c"
        self.rgb_grey_id_dict_manager("#8a364c", "pizza")

    def donut_fg(self):
        self.color_fg = "#8c6e28"
        self.rgb_grey_id_dict_manager("#8c6e28", "donut")

    def cake_fg(self):
        self.color_fg = "#e46085"
        self.rgb_grey_id_dict_manager("#e46085", "cake")

    def chair_fg(self):
        self.color_fg = "#6c2818"
        self.rgb_grey_id_dict_manager("#6c2818", "chair")

    def couch_fg(self):
        self.color_fg = "#f2d3e9"
        self.rgb_grey_id_dict_manager("#f2d3e9", "couch")

    def potted_plant_fg(self):
        self.color_fg = "#62d50f"
        self.rgb_grey_id_dict_manager("#62d50f", "potted plant")

    def bed_fg(self):
        self.color_fg = "#f2d3e9"
        self.rgb_grey_id_dict_manager("#f2d3e9", "bed")

    def mirror_stuff_fg(self):
        self.color_fg = "#87a5cd"
        self.rgb_grey_id_dict_manager("#87a5cd", "mirror-stuff")

    def dining_table_fg(self):
        self.color_fg = "#d7c391"
        self.rgb_grey_id_dict_manager("#d7c391", "dining table")

    def window_fg(self):
        self.color_fg = "#665988"
        self.rgb_grey_id_dict_manager("#665988", "window")

    def desk_fg(self):
        self.color_fg = "#aa4967"
        self.rgb_grey_id_dict_manager("#aa4967", "desk")

    def toilet_fg(self):
        self.color_fg = "#7b31ed"
        self.rgb_grey_id_dict_manager("#7b31ed", "toilet")

    def door_fg(self):
        self.color_fg = "#fe4446"
        self.rgb_grey_id_dict_manager("#fe4446", "door")

    def tv_fg(self):
        self.color_fg = "#0a365a"
        self.rgb_grey_id_dict_manager("#0a365a", "tv")

    def laptop_fg(self):
        self.color_fg = "#bcdfd7"
        self.rgb_grey_id_dict_manager("#bcdfd7", "laptop")

    def mouse_fg(self):
        self.color_fg = "#4749a4"
        self.rgb_grey_id_dict_manager("#4749a4", "mouse")

    def remote_fg(self):
        self.color_fg = "#dc308c"
        self.rgb_grey_id_dict_manager("#dc308c", "remote")

    def keyboard_fg(self):
        self.color_fg = "#d28535"
        self.rgb_grey_id_dict_manager("#d28535", "keyboard")

    def cell_phone_fg(self):
        self.color_fg = "#eb69b6"
        self.rgb_grey_id_dict_manager("#eb69b6", "cell phone")

    def microwave_fg(self):
        self.color_fg = "#494171"
        self.rgb_grey_id_dict_manager("#494171", "microwave")

    def oven_fg(self):
        self.color_fg = "#174f46"
        self.rgb_grey_id_dict_manager("#174f46", "oven")

    def toaster_fg(self):
        self.color_fg = "#1c575f"
        self.rgb_grey_id_dict_manager("#1c575f", "toaster")

    def sink_fg(self):
        self.color_fg = "#ddf289"
        self.rgb_grey_id_dict_manager("#ddf289", "sink")

    def refrigerator_fg(self):
        self.color_fg = "#9d68bd"
        self.rgb_grey_id_dict_manager("#9d68bd", "refrigerator")

    def blender_fg(self):
        self.color_fg = "#9f714a"
        self.rgb_grey_id_dict_manager("#9f714a", "blender")

    def book_fg(self):
        self.color_fg = "#715fcc"
        self.rgb_grey_id_dict_manager("#715fcc", "book")

    def clock_fg(self):
        self.color_fg = "#6c9617"
        self.rgb_grey_id_dict_manager("#6c9617", "clock")

    def vase_fg(self):
        self.color_fg = "#f78adb"
        self.rgb_grey_id_dict_manager("#f78adb", "vase")

    def scissors_fg(self):
        self.color_fg = "#ee3bb3"
        self.rgb_grey_id_dict_manager("#ee3bb3", "scissors")

    def teddy_bear_fg(self):
        self.color_fg = "#9f714a"
        self.rgb_grey_id_dict_manager("#9f714a", "teddy bear")

    def hair_drier_fg(self):
        self.color_fg = "#f014a1"
        self.rgb_grey_id_dict_manager("#f014a1", "hair drier")

    def toothbrush_fg(self):
        self.color_fg = "#74dbfc"
        self.rgb_grey_id_dict_manager("#74dbfc", "toothbrush")

    def hair_brush_fg(self):  # last class of Thing
        self.color_fg = "#dd7cc3"
        self.rgb_grey_id_dict_manager("#dd7cc3", "hair brush")

    def banner_fg(self):  # first class of Stuff
        self.color_fg = "#6c9617"
        self.rgb_grey_id_dict_manager("#6c9617", "banner")

    def blanket_fg(self):
        self.color_fg = "#bdc8a2"
        self.rgb_grey_id_dict_manager("#bdc8a2", "blanket")

    def branch_fg(self):
        self.color_fg = "#584e26"
        self.rgb_grey_id_dict_manager("#584e26", "branch")

    def building_other_fg(self):
        self.color_fg = "#584f2f"
        self.rgb_grey_id_dict_manager("#584f2f", "building-other")

    def bush_fg(self):
        self.color_fg = "#0e8809"
        self.rgb_grey_id_dict_manager("#0e8809", "bush")

    def cabinet_fg(self):
        self.color_fg = "#103dae"
        self.rgb_grey_id_dict_manager("#103dae", "cabinet")

    def cage_fg(self):
        self.color_fg = "#c09003"
        self.rgb_grey_id_dict_manager("#c09003", "cage")

    def cardboard_fg(self):
        self.color_fg = "#1d3c07"
        self.rgb_grey_id_dict_manager("#1d3c07", "cardboard")

    def carpet_fg(self):
        self.color_fg = "#d55509"
        self.rgb_grey_id_dict_manager("#d55509", "carpet")

    def ceiling_other_fg(self):
        self.color_fg = "#d0f790"
        self.rgb_grey_id_dict_manager("#d0f790", "ceiling-other")

    def ceiling_tile_fg(self):
        self.color_fg = "#4fe78c"
        self.rgb_grey_id_dict_manager("#4fe78c", "ceiling-tile")

    def cloth_fg(self):
        self.color_fg = "#ed5f29"
        self.rgb_grey_id_dict_manager("#ed5f29", "cloth")

    def clothes_fg(self):
        self.color_fg = "#a55462"
        self.rgb_grey_id_dict_manager("#a55462", "clothes")

    def clouds_fg(self):
        self.color_fg = "#74dbfe"
        self.rgb_grey_id_dict_manager("#74dbfe", "clouds")

    def counter_fg(self):
        self.color_fg = "#025622"
        self.rgb_grey_id_dict_manager("#025622", "counter")

    def cupboard_fg(self):
        self.color_fg = "#76fa08"
        self.rgb_grey_id_dict_manager("#76fa08", "cupboard")

    def curtain_fg(self):
        self.color_fg = "#aa4967"
        self.rgb_grey_id_dict_manager("#aa4967", "curtain")

    def desk_stuff_fg(self):
        self.color_fg = "#e4db52"
        self.rgb_grey_id_dict_manager("#e4db52", "desk-stuff")

    def dirt_fg(self):
        self.color_fg = "#7fae9e"
        self.rgb_grey_id_dict_manager("#7fae9e", "dirt")

    def door_stuff_fg(self):
        self.color_fg = "#7b12b7"
        self.rgb_grey_id_dict_manager("#7b12b7", "door-stuff")

    def fence_fg(self):
        self.color_fg = "#b7bf62"
        self.rgb_grey_id_dict_manager("#b7bf62", "fence")

    def floor_marble_fg(self):
        self.color_fg = "#0c2a5f"
        self.rgb_grey_id_dict_manager("#0c2a5f", "floor-marble")

    def floor_other_fg(self):
        self.color_fg = "#657b9c"
        self.rgb_grey_id_dict_manager("#657b9c", "floor-other")

    def floor_stone_fg(self):
        self.color_fg = "#7aa3ea"
        self.rgb_grey_id_dict_manager("#7aa3ea", "floor-stone")

    def floor_tile_fg(self):
        self.color_fg = "#b2fbc7"
        self.rgb_grey_id_dict_manager("#b2fbc7", "floor-tile")

    def floor_wood_fg(self):
        self.color_fg = "#7fa54e"
        self.rgb_grey_id_dict_manager("#7fa54e", "floor-wood")

    def flower_fg(self):
        self.color_fg = "#fd727c"
        self.rgb_grey_id_dict_manager("#fd727c", "flower")

    def fog_fg(self):
        self.color_fg = "#7fae9e"
        self.rgb_grey_id_dict_manager("#7fae9e", "fog")

    def food_other_fg(self):
        self.color_fg = "#942bca"
        self.rgb_grey_id_dict_manager("#942bca", "food-other")

    def fruit_fg(self):
        self.color_fg = "#c7510e"
        self.rgb_grey_id_dict_manager("#c7510e", "fruit")

    def furniture_other_fg(self):
        self.color_fg = "#5a060b"
        self.rgb_grey_id_dict_manager("#5a060b", "furniture-other")

    def grass_fg(self):
        self.color_fg = "#38700c"
        self.rgb_grey_id_dict_manager("#38700c", "grass")

    def gravel_fg(self):
        self.color_fg = "#0b3d6e"
        self.rgb_grey_id_dict_manager("#0b3d6e", "gravel")

    def ground_other_fg(self):
        self.color_fg = "#ad741a"
        self.rgb_grey_id_dict_manager("#ad741a", "ground-other")

    def hill_fg(self):
        self.color_fg = "#bcef69"
        self.rgb_grey_id_dict_manager("#bcef69", "hill")

    def house_fg(self):
        self.color_fg = "#e95974"
        self.rgb_grey_id_dict_manager("#e95974", "house")

    def leaves_fg(self):
        self.color_fg = "#25d626"
        self.rgb_grey_id_dict_manager("#25d626", "leaves")

    def light_fg(self):
        self.color_fg = "#f5f222"
        self.rgb_grey_id_dict_manager("#f5f222", "light")

    def mat_fg(self):
        self.color_fg = "#0723a5"
        self.rgb_grey_id_dict_manager("#0723a5", "mat")

    def metal_fg(self):
        self.color_fg = "#657b9c"
        self.rgb_grey_id_dict_manager("#657b9c", "metal")

    def moss_fg(self):
        self.color_fg = "#25d626"
        self.rgb_grey_id_dict_manager("#25d626", "moss")

    def mountain_fg(self):
        self.color_fg = "#76b7b2"
        self.rgb_grey_id_dict_manager("#76b7b2", "mountain")

    def mud_fg(self):
        self.color_fg = "#4a351a"
        self.rgb_grey_id_dict_manager("#4a351a", "mud")

    def napkin_fg(self):
        self.color_fg = "#837acb"
        self.rgb_grey_id_dict_manager("#837acb", "napkin")

    def net_fg(self):
        self.color_fg = "#c7510e"
        self.rgb_grey_id_dict_manager("#c7510e", "net")

    def paper_fg(self):
        self.color_fg = "#b6ccbe"
        self.rgb_grey_id_dict_manager("#b6ccbe", "paper")

    def pavement_fg(self):
        self.color_fg = "#a7abc2"
        self.rgb_grey_id_dict_manager("#a7abc2", "pavement")

    def pillow_fg(self):
        self.color_fg = "#ac6dd0"
        self.rgb_grey_id_dict_manager("#ac6dd0", "pillow")

    def plant_other_fg(self):
        self.color_fg = "#149688"
        self.rgb_grey_id_dict_manager("#149688", "plant-other")

    def plastic_fg(self):
        self.color_fg = "#6c5351"
        self.rgb_grey_id_dict_manager("#6c5351", "plastic")

    def platform_fg(self):
        self.color_fg = "#fe46f6"
        self.rgb_grey_id_dict_manager("#fe46f6", "platform")

    def playingfield_fg(self):
        self.color_fg = "#9e0b30"
        self.rgb_grey_id_dict_manager("#9e0b30", "playingfield")

    def railing_fg(self):
        self.color_fg = "#4d0d98"
        self.rgb_grey_id_dict_manager("#4d0d98", "railing")

    def railroad_fg(self):
        self.color_fg = "#7dd8d9"
        self.rgb_grey_id_dict_manager("#7dd8d9", "railroad")

    def river_fg(self):
        self.color_fg = "#7653fd"
        self.rgb_grey_id_dict_manager("#7653fd", "river")

    def road_fg(self):
        self.color_fg = "#4d0c3d"
        self.rgb_grey_id_dict_manager("#4d0c3d", "road")

    def rock_fg(self):
        self.color_fg = "#81656d"
        self.rgb_grey_id_dict_manager("#81656d", "rock")

    def roof_fg(self):
        self.color_fg = "#0e471e"
        self.rgb_grey_id_dict_manager("#0e471e", "roof")

    def rug_fg(self):
        self.color_fg = "#fe46f6"
        self.rgb_grey_id_dict_manager("#fe46f6", "rug")

    def salad_fg(self):
        self.color_fg = "#8b5e61"
        self.rgb_grey_id_dict_manager("#8b5e61", "salad")

    def sand_fg(self):
        self.color_fg = "#dbc386"
        self.rgb_grey_id_dict_manager("#dbc386", "sand")

    def sea_fg(self):
        self.color_fg = "#10ebf2"
        self.rgb_grey_id_dict_manager("#10ebf2", "sea")

    def shelf_fg(self):
        self.color_fg = "#65731f"
        self.rgb_grey_id_dict_manager("#65731f", "shelf")

    def skyscraper_fg(self):
        self.color_fg = "#9d1670"
        self.rgb_grey_id_dict_manager("#9d1670", "skyscraper")

    def snow_fg(self):
        self.color_fg = "#ecd9fe"
        self.rgb_grey_id_dict_manager("#ecd9fe", "snow")

    def solid_other_fg(self):
        self.color_fg = "#05b8dc"
        self.rgb_grey_id_dict_manager("#05b8dc", "solid-other")

    def stairs_fg(self):
        self.color_fg = "#8f0905"
        self.rgb_grey_id_dict_manager("#8f0905", "stairs")

    def stone_fg(self):
        self.color_fg = "#a7abc2"
        self.rgb_grey_id_dict_manager("#a7abc2", "stone")

    def straw_fg(self):
        self.color_fg = "#4cad63"
        self.rgb_grey_id_dict_manager("#4cad63", "straw")

    def structural_other_fg(self):
        self.color_fg = "#b6ccbe"
        self.rgb_grey_id_dict_manager("#b6ccbe", "structural-other")

    def table_fg(self):
        self.color_fg = "#9d8356"
        self.rgb_grey_id_dict_manager("#9d8356", "table")

    def tent_fg(self):
        self.color_fg = "#b519e7"
        self.rgb_grey_id_dict_manager("#b519e7", "tent")

    def textile_other_fg(self):
        self.color_fg = "#f91c1c"
        self.rgb_grey_id_dict_manager("#f91c1c", "textile-other")

    def towel_fg(self):
        self.color_fg = "#4625f5"
        self.rgb_grey_id_dict_manager("#4625f5", "towel")

    def vegetable_fg(self):
        self.color_fg = "#b96fd4"
        self.rgb_grey_id_dict_manager("#b96fd4", "vegetable")

    def wall_concrete_fg(self):
        self.color_fg = "#3a364c"
        self.rgb_grey_id_dict_manager("#3a364c", "wall-concrete")

    def wall_brick_fg(self):
        self.color_fg = "#151f09"
        self.rgb_grey_id_dict_manager("#151f09", "wall-brick")

    def wall_other_fg(self):
        self.color_fg = "#ab11b5"
        self.rgb_grey_id_dict_manager("#ab11b5", "wall-other")

    def wall_wood_fg(self):
        self.color_fg = "#978f48"
        self.rgb_grey_id_dict_manager("#978f48", "wall-wood")

    def wall_panel_fg(self):
        self.color_fg = "#65ef40"
        self.rgb_grey_id_dict_manager("#65ef40", "wall-panel")

    def wall_stone_fg(self):
        self.color_fg = "#8fada7"
        self.rgb_grey_id_dict_manager("#8fada7", "wall-stone")

    def wall_tile_fg(self):
        self.color_fg = "#3b2b64"
        self.rgb_grey_id_dict_manager("#3b2b64", "wall-tile")

    def water_other_fg(self):
        self.color_fg = "#51d2ec"
        self.rgb_grey_id_dict_manager("#51d2ec", "water-other")

    def waterdrops_fg(self):
        self.color_fg = "#05b8dc"
        self.rgb_grey_id_dict_manager("#05b8dc", "waterdrops")

    def window_blind_fg(self):
        self.color_fg = "#d3476e"
        self.rgb_grey_id_dict_manager("#d3476e", "window-blind")

    def window_other_fg(self):
        self.color_fg = "#dfe39d"
        self.rgb_grey_id_dict_manager("#dfe39d", "window-other")

    def wood_fg(self):
        self.color_fg = "#743975"
        self.rgb_grey_id_dict_manager("#743975", "wood")

    def airplane_fg(self):
        self.color_fg = "#fd6210"
        self.rgb_grey_id_dict_manager("#fd6210", "airplane")

    def mirror_fg(self):
        self.color_fg = "#55ebd2"
        self.rgb_grey_id_dict_manager("#55ebd2", "mirror")

    # def load_picture(self):
    #   path = filedialog.askdirectory(parent=root, title="Choose an image...")
    #  img = ImageTk.PhotoImage(Image.open(path))
    # self.c.create_image(1000, 1000, image=img, anchor=NW)

    def save_as_rgb_image(self, path):
        canvas = self.c.postscript(colormode="color")
        img = np.array(Image.open(io.BytesIO(canvas.encode("utf-8"))))
        img = cv2.resize(img, (256, 256))
        rgb_image = np.zeros((img.shape[0], img.shape[1], 3))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                pixel = img[i][j]
                if int(pixel[0]) == 255 & int(pixel[1]) == 255 & int(pixel[2]) == 255:
                    rgb_image[i][j] = (116, 219, 254)
                else:
                    rgb_image[i][j] = img[i][j]
        if path:
            skimage.io.imsave(path + ".png", rgb_image)
        else:
            ts = time.time()
            time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            skimage.io.imsave("/home/bernihoh/Bachelor/GUI-test-pictures/" + time_stamp + ".png", np.uint8(rgb_image))
        return

    def save_as_grey_image(self, path):
        canvas = self.c.postscript(colormode="color")
        img = np.asarray(Image.open(io.BytesIO(canvas.encode("utf-8"))))
        grey_img = np.zeros((img.shape[0], img.shape[1]))
        for i in range(img.shape[1]):
            for j in range(img.shape[0]):
                try:
                    grey_img[j][i] = self.rgb_grey_id_dict[tuple(img[j][i])]
                except:
                    pass
        grey_img = grey_img.astype(np.ubyte)
        if path:
            skimage.io.imsave(path + ".png", grey_img)
        else:
            ts = time.time()
            time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            # skimage.io.imsave("/home/bernihoh/Bachelor/GUI-test-pictures/"+time_stamp+".png", grey_img)
            skimage.io.imsave(
                "/home/bernihoh/Bachelor/SMS/MaskRCNN/samples/SMSNetworks/head_place_detection/pic_train/" + time_stamp + ".png",
                grey_img)
        return

    def save_label_map(self, mode='RGB'):
        file_explorer = root
        file_explorer.option_add('*foreground', 'black')
        file_explorer.option_add('*activeForeground', 'black')
        style = ttk.Style(file_explorer)
        style.configure('TLabel', foreground='black')
        style.configure('TEntry', foreground='black')
        style.configure('TMenubutton', foreground='black')
        style.configure('TButton', foreground='black')
        path = filedialog.askdirectory(parent=root, title="Select directory...")
        ts = time.time()
        time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        if mode == 'RGB':
            self.save_as_rgb_image(path + '/' + time_stamp)
        elif mode == 'GRAY':
            self.save_as_grey_image(path + '/' + time_stamp)
        else:
            return
        return

    def save_last_synth_img(self):
        if not self.last_synth_img is None:
            file_explorer = root
            file_explorer.option_add('*foreground', 'black')
            file_explorer.option_add('*activeForeground', 'black')
            style = ttk.Style(file_explorer)
            style.configure('TLabel', foreground='black')
            style.configure('TEntry', foreground='black')
            style.configure('TMenubutton', foreground='black')
            style.configure('TButton', foreground='black')
            path = filedialog.askdirectory(parent=root, title="Save synthesized image...")
            ts = time.time()
            time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            img_path = path + "/" + time_stamp + ".png"
            skimage.io.imsave(img_path, self.last_synth_img)
        return

    def save_last_SPADE_label(self):
        if not self.last_SPADE_label is None:
            file_explorer = root
            file_explorer.option_add('*foreground', 'black')
            file_explorer.option_add('*activeForeground', 'black')
            style = ttk.Style(file_explorer)
            style.configure('TLabel', foreground='black')
            style.configure('TEntry', foreground='black')
            style.configure('TMenubutton', foreground='black')
            style.configure('TButton', foreground='black')
            path = filedialog.askdirectory(parent=root, title="Save SPADE label...")
            ts = time.time()
            time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            img_path = path + "/" + time_stamp + ".png"
            skimage.io.imsave(img_path, self.last_SPADE_label)
        return

    def detect_objects(self):
        file_explorer = root
        file_explorer.option_add('*foreground', 'black')
        file_explorer.option_add('*activeForeground', 'black')
        style = ttk.Style(file_explorer)
        style.configure('TLabel', foreground='black')
        style.configure('TEntry', foreground='black')
        style.configure('TMenubutton', foreground='black')
        style.configure('TButton', foreground='black')
        path = filedialog.askdirectory(parent=root, title="Choose an image...")
        if path:
            obj_det = object_detection()
            img = object_detection.detect(obj_det, path, 127)
            ts = time.time()
            time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            # skimage.io.imsave("/home/bernihoh/Bachelor/GUI-test-pictures/"+time_stamp+".png", img)
            skimage.io.imsave(
                "/home/bernihoh/Bachelor/SMS/MaskRCNN/samples/SMSNetworks/head_place_detection/pic_train/" + time_stamp + ".png",
                img)
        return

    def generate_picture(self):

        ts = time.time()
        time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        val_img_path = "/home/bernihoh/Bachelor/SPADE/datasets/coco_stuff/val_img/" + time_stamp

        val_inst_path = "/home/bernihoh/Bachelor/SPADE/datasets/coco_stuff/val_inst/" + time_stamp

        val_label_path = "/home/bernihoh/Bachelor/SPADE/datasets/coco_stuff/val_label/" + time_stamp

        self.save_as_rgb_image(val_img_path)
        self.save_as_grey_image(val_inst_path)
        self.save_as_grey_image(val_label_path)
        label, img = pic_gen.generate_1()
        head_creator = head_creation(label, img)
        manipulated_img = head_creator.head_creator()
        wall_decorator = decorate_walls(manipulated_img, label)
        manipulated_img = wall_decorator.decorate_walls()
        #manipulated_img = img
        self.last_synth_img = manipulated_img
        self.last_SPADE_label = label
        plt.imshow(manipulated_img)
        plt.show()
        window = tkinter.Toplevel()
        window.title('Synthesized Picture')
        img = ImageTk.PhotoImage(Image.fromarray(manipulated_img))
        panel = tkinter.Label(window, image=img)
        panel.pack(side="bottom", fill="both", expand="yes")
        window.mainloop()
        print("synthesized")
        return

    def drawWidgets(self):
        self.controls = Frame(self.master, padx=5, pady=5)
        Label(self.controls, text='Pen Width:', font=('arial 18')).grid(row=0, column=0)
        self.slider = ttk.Scale(self.controls, from_=1, to=150, command=self.changeW, orient=VERTICAL)
        self.slider.set(self.penwidth)
        self.slider.grid(row=0, column=1, ipadx=30)
        self.controls.pack(side=LEFT)

        self.c = Canvas(self.master, width=256, height=256, bg=self.color_bg)
        self.c.pack(fill=BOTH, expand=True)

        menu = Menu(self.master)
        self.master.config(menu=menu)
        # filemenu = Menu(menu)
        # colormenu = Menu(menu)
        plants_animals = Menu(menu)
        outdoor = Menu(menu)
        indoor = Menu(menu)
        food = Menu(menu)
        other = Menu(menu)
        buildings_constructs = Menu(menu)
        vehicle_roads = Menu(menu)
        sports_hobbies = Menu(menu)
        kitchen = Menu(menu)
        furniture_establishment = Menu(menu)
        plants_animals.add_command(label="tree", command=lambda: self.tree_fg())
        other.add_command(label="person", command=lambda: self.person_fg())
        buildings_constructs.add_command(label="bridge", command=lambda: self.bridge_fg())
        outdoor.add_command(label="sky-other", command=lambda: self.sky_other_fg())
        vehicle_roads.add_command(label="bicycle", command=lambda: self.bicycle_fg())
        vehicle_roads.add_command(label="car", command=lambda: self.car_fg())
        vehicle_roads.add_command(label="motorcycle", command=lambda: self.motorcycle_fg())
        vehicle_roads.add_command(label="airplane", command=lambda: self.airplane_fg())
        vehicle_roads.add_command(label="bus", command=lambda: self.bus_fg())
        vehicle_roads.add_command(label="train", command=lambda: self.train_fg())
        vehicle_roads.add_command(label="truck", command=lambda: self.truck_fg())
        vehicle_roads.add_command(label="boat", command=lambda: self.boat_fg())
        vehicle_roads.add_command(label="traffic light", command=lambda: self.traffic_light_fg())
        vehicle_roads.add_command(label="fire hydrant", command=lambda: self.fire_hydrant_fg())
        vehicle_roads.add_command(label="street sign", command=lambda: self.street_sign_fg())
        vehicle_roads.add_command(label="stop sign", command=lambda: self.stop_sign_fg())
        vehicle_roads.add_command(label="parking meter", command=lambda: self.parking_meter_fg())
        plants_animals.add_command(label="bench", command=lambda: self.bench_fg())
        plants_animals.add_command(label="bird", command=lambda: self.bird_fg())
        plants_animals.add_command(label="cat", command=lambda: self.cat_fg())
        plants_animals.add_command(label="dog", command=lambda: self.dog_fg())
        plants_animals.add_command(label="horse", command=lambda: self.horse_fg())
        plants_animals.add_command(label="sheep", command=lambda: self.sheep_fg())
        plants_animals.add_command(label="cow", command=lambda: self.cow_fg())
        plants_animals.add_command(label="elephant", command=lambda: self.elephant_fg())
        plants_animals.add_command(label="bear", command=lambda: self.bear_fg())
        plants_animals.add_command(label="zebra", command=lambda: self.zebra_fg())
        other.add_command(label="hat", command=lambda: self.hat_fg())
        plants_animals.add_command(label="giraffe", command=lambda: self.giraffe_fg())
        other.add_command(label="backpack", command=lambda: self.backpack_fg())
        other.add_command(label="umbrella", command=lambda: self.umbrella_fg())
        other.add_command(label="shoe", command=lambda: self.shoe_fg())
        other.add_command(label="eye glasses", command=lambda: self.eye_glasses_fg())
        other.add_command(label="handbag", command=lambda: self.handbag_fg())
        other.add_command(label="tie", command=lambda: self.tie_fg())
        other.add_command(label="suitcase", command=lambda: self.suitcase_fg())
        sports_hobbies.add_command(label="frisbee", command=lambda: self.frisbee_fg())
        sports_hobbies.add_command(label="skis", command=lambda: self.skis_fg())
        sports_hobbies.add_command(label="snowboard", command=lambda: self.snowboard_fg())
        sports_hobbies.add_command(label="sports ball", command=lambda: self.sports_ball_fg())
        sports_hobbies.add_command(label="kite", command=lambda: self.kite_fg())
        sports_hobbies.add_command(label="baseball bat", command=lambda: self.baseball_bat_fg())
        sports_hobbies.add_command(label="baseball glove", command=lambda: self.baseball_glove_fg())
        sports_hobbies.add_command(label="skateboard", command=lambda: self.skateboard_fg())
        sports_hobbies.add_command(label="surfboard", command=lambda: self.surfboard_fg())
        sports_hobbies.add_command(label="tennis racket", command=lambda: self.tennis_racket_fg())
        kitchen.add_command(label="bottle", command=lambda: self.bottle_fg())
        kitchen.add_command(label="plate", command=lambda: self.plate_fg())
        kitchen.add_command(label="wine glass", command=lambda: self.wine_glass_fg())
        kitchen.add_command(label="cup", command=lambda: self.cup_fg())
        kitchen.add_command(label="fork", command=lambda: self.fork_fg())
        kitchen.add_command(label="knife", command=lambda: self.knife_fg())
        kitchen.add_command(label="spoon", command=lambda: self.spoon_fg())
        kitchen.add_command(label="bowl", command=lambda: self.bowl_fg())
        food.add_command(label="apple", command=lambda: self.apple_fg())
        food.add_command(label="orange", command=lambda: self.orange_fg())
        food.add_command(label="sandwich", command=lambda: self.sandwich_fg())
        food.add_command(label="banana", command=lambda: self.banana_fg())
        food.add_command(label="broccoli", command=lambda: self.broccoli_fg())
        food.add_command(label="carrot", command=lambda: self.carrot_fg())
        food.add_command(label="hot dog", command=lambda: self.hot_dog_fg())
        food.add_command(label="pizza", command=lambda: self.pizza_fg())
        food.add_command(label="donut", command=lambda: self.donut_fg())
        food.add_command(label="cake", command=lambda: self.cake_fg())
        buildings_constructs.add_command(label="window", command=lambda: self.window_fg())
        furniture_establishment.add_command(label="chair", command=lambda: self.chair_fg())
        furniture_establishment.add_command(label="couch", command=lambda: self.couch_fg())
        plants_animals.add_command(label="potted plant", command=lambda: self.potted_plant_fg())
        furniture_establishment.add_command(label="desk", command=lambda: self.desk_fg())
        furniture_establishment.add_command(label="bed", command=lambda: self.bed_fg())
        furniture_establishment.add_command(label="mirror", command=lambda: self.mirror_fg())
        kitchen.add_command(label="dining table", command=lambda: self.dining_table_fg())
        furniture_establishment.add_command(label="toilet", command=lambda: self.toilet_fg())
        furniture_establishment.add_command(label="door", command=lambda: self.door_fg())
        indoor.add_command(label="tv", command=lambda: self.tv_fg())
        indoor.add_command(label="laptop", command=lambda: self.laptop_fg())
        indoor.add_command(label="mouse", command=lambda: self.mouse_fg())
        indoor.add_command(label="keyboard", command=lambda: self.keyboard_fg())
        indoor.add_command(label="remote", command=lambda: self.remote_fg())
        indoor.add_command(label="cell phone", command=lambda: self.cell_phone_fg())
        kitchen.add_command(label="oven", command=lambda: self.oven_fg())
        kitchen.add_command(label="microwave", command=lambda: self.microwave_fg())
        kitchen.add_command(label="toaster", command=lambda: self.toaster_fg())
        kitchen.add_command(label="sink", command=lambda: self.sink_fg())
        kitchen.add_command(label="refrigerator", command=lambda: self.refrigerator_fg())
        kitchen.add_command(label="blender", command=lambda: self.blender_fg())
        indoor.add_command(label="book", command=lambda: self.book_fg())
        indoor.add_command(label="clock", command=lambda: self.clock_fg())
        furniture_establishment.add_command(label="vase", command=lambda: self.vase_fg())
        indoor.add_command(label="scissors", command=lambda: self.scissors_fg())
        indoor.add_command(label="hair drier", command=lambda: self.hair_drier_fg())
        indoor.add_command(label="teddy bear", command=lambda: self.teddy_bear_fg())
        indoor.add_command(label="toothbrush", command=lambda: self.toothbrush_fg())
        indoor.add_command(label="hair brush", command=lambda: self.hair_brush_fg())
        furniture_establishment.add_command(label="banner", command=lambda: self.banner_fg())
        plants_animals.add_command(label="branch", command=lambda: self.branch_fg())
        indoor.add_command(label="blanket", command=lambda: self.blanket_fg())
        buildings_constructs.add_command(label="building-other", command=lambda: self.building_other_fg())
        plants_animals.add_command(label="bush", command=lambda: self.bush_fg())
        furniture_establishment.add_command(label="cabinet", command=lambda: self.cabinet_fg())
        furniture_establishment.add_command(label="cage", command=lambda: self.cage_fg())
        furniture_establishment.add_command(label="cardboard", command=lambda: self.cardboard_fg())
        furniture_establishment.add_command(label="carpet", command=lambda: self.carpet_fg())
        buildings_constructs.add_command(label="ceiling-other", command=lambda: self.ceiling_other_fg())
        buildings_constructs.add_command(label="ceiling-tile", command=lambda: self.ceiling_tile_fg())
        other.add_command(label="cloth", command=lambda: self.cloth_fg())
        other.add_command(label="clothes", command=lambda: self.clothes_fg())
        outdoor.add_command(label="clouds", command=lambda: self.clouds_fg())
        indoor.add_command(label="counter", command=lambda: self.counter_fg())
        furniture_establishment.add_command(label="cupboard", command=lambda: self.cupboard_fg())
        indoor.add_command(label="curtain", command=lambda: self.curtain_fg())
        other.add_command(label="desk-stuff", command=lambda: self.desk_stuff_fg())
        outdoor.add_command(label="dirt", command=lambda: self.dirt_fg())
        buildings_constructs.add_command(label="door-stuff", command=lambda: self.door_stuff_fg())
        buildings_constructs.add_command(label="fence", command=lambda: self.fence_fg())
        furniture_establishment.add_command(label="floor-marble", command=lambda: self.floor_marble_fg())
        furniture_establishment.add_command(label="floor-other", command=lambda: self.floor_other_fg())
        furniture_establishment.add_command(label="floor-stone", command=lambda: self.floor_stone_fg())
        furniture_establishment.add_command(label="floor-tile", command=lambda: self.floor_tile_fg())
        furniture_establishment.add_command(label="floor-wood", command=lambda: self.floor_wood_fg())
        plants_animals.add_command(label="flower", command=lambda: self.flower_fg())
        outdoor.add_command(label="fog", command=lambda: self.fog_fg())
        food.add_command(label="food-other", command=lambda: self.food_other_fg())
        food.add_command(label="fruit", command=lambda: self.fruit_fg())
        furniture_establishment.add_command(label="furniture-other", command=lambda: self.furniture_other_fg())
        plants_animals.add_command(label="grass", command=lambda: self.grass_fg())
        outdoor.add_command(label="gravel", command=lambda: self.gravel_fg())
        outdoor.add_command(label="ground-other", command=lambda: self.ground_other_fg())
        outdoor.add_command(label="hill", command=lambda: self.hill_fg())
        plants_animals.add_command(label="leaves", command=lambda: self.leaves_fg())
        outdoor.add_command(label="light", command=lambda: self.light_fg())
        furniture_establishment.add_command(label="mat", command=lambda: self.mat_fg())
        other.add_command(label="metal", command=lambda: self.metal_fg())
        indoor.add_command(label="mirror-stuff", command=lambda: self.mirror_stuff_fg())
        plants_animals.add_command(label="moss", command=lambda: self.moss_fg())
        outdoor.add_command(label="mountain", command=lambda: self.mountain_fg())
        outdoor.add_command(label="mud", command=lambda: self.mud_fg())
        other.add_command(label="napkin", command=lambda: self.napkin_fg())
        other.add_command(label="net", command=lambda: self.net_fg())
        indoor.add_command(label="paper", command=lambda: self.paper_fg())
        vehicle_roads.add_command(label="pavement", command=lambda: self.pavement_fg())
        indoor.add_command(label="pillow", command=lambda: self.pillow_fg())
        plants_animals.add_command(label="plant-other", command=lambda: self.plant_other_fg())
        other.add_command(label="plastic", command=lambda: self.plastic_fg())
        sports_hobbies.add_command(label="playingfield", command=lambda: self.playingfield_fg())
        buildings_constructs.add_command(label="platform", command=lambda: self.platform_fg())
        buildings_constructs.add_command(label="railing", command=lambda: self.railing_fg())
        vehicle_roads.add_command(label="railroad", command=lambda: self.railroad_fg())
        outdoor.add_command(label="river", command=lambda: self.river_fg())
        vehicle_roads.add_command(label="road", command=lambda: self.road_fg())
        outdoor.add_command(label="rock", command=lambda: self.rock_fg())
        buildings_constructs.add_command(label="roof", command=lambda: self.roof_fg())
        indoor.add_command(label="rug", command=lambda: self.rug_fg())
        food.add_command(label="salad", command=lambda: self.salad_fg())
        outdoor.add_command(label="sand", command=lambda: self.sand_fg())
        outdoor.add_command(label="sea", command=lambda: self.sea_fg())
        furniture_establishment.add_command(label="shelf", command=lambda: self.shelf_fg())
        buildings_constructs.add_command(label="skyscraper", command=lambda: self.skyscraper_fg())
        outdoor.add_command(label="snow", command=lambda: self.snow_fg())
        buildings_constructs.add_command(label="solid-other", command=lambda: self.solid_other_fg())
        furniture_establishment.add_command(label="stairs", command=lambda: self.stairs_fg())
        outdoor.add_command(label="stone", command=lambda: self.stone_fg())
        other.add_command(label="straw", command=lambda: self.straw_fg())
        buildings_constructs.add_command(label="structural-other", command=lambda: self.structural_other_fg())
        furniture_establishment.add_command(label="table", command=lambda: self.table_fg())
        buildings_constructs.add_command(label="tent", command=lambda: self.tent_fg())
        other.add_command(label="textile-other", command=lambda: self.textile_other_fg())
        other.add_command(label="towel", command=lambda: self.towel_fg())
        food.add_command(label="vegetable", command=lambda: self.vegetable_fg())
        buildings_constructs.add_command(label="wall-brick", command=lambda: self.wall_brick_fg())
        buildings_constructs.add_command(label="wall-concrete", command=lambda: self.wall_concrete_fg())
        buildings_constructs.add_command(label="wall-other", command=lambda: self.wall_other_fg())
        buildings_constructs.add_command(label="wall-panel", command=lambda: self.wall_panel_fg())
        buildings_constructs.add_command(label="wall-stone", command=lambda: self.wall_stone_fg())
        buildings_constructs.add_command(label="wall-tile", command=lambda: self.wall_tile_fg())
        buildings_constructs.add_command(label="wall-wood", command=lambda: self.wall_wood_fg())
        outdoor.add_command(label="water-other", command=lambda: self.water_other_fg())
        outdoor.add_command(label="waterdrops", command=lambda: self.waterdrops_fg())
        buildings_constructs.add_command(label="window-blind", command=lambda: self.window_blind_fg())
        buildings_constructs.add_command(label="window-other", command=lambda: self.window_other_fg())
        outdoor.add_command(label="wood", command=lambda: self.wood_fg())

        ## menu.add_cascade(label="Change Class", menu=class_chooser)
        menu.add_cascade(label="Buildings & Constructs", menu=buildings_constructs)
        menu.add_cascade(label="Outdoor", menu=outdoor)
        menu.add_cascade(label="Indoor", menu=indoor)
        menu.add_cascade(label="Furniture & Establishment", menu=furniture_establishment)
        menu.add_cascade(label="Sports & Hobbies", menu=sports_hobbies)
        menu.add_cascade(label="Kitchen", menu=kitchen)
        menu.add_cascade(label="Food", menu=food)
        menu.add_cascade(label="Plants & Animals", menu=plants_animals)
        menu.add_cascade(label="Other", menu=other)
        # menu.add_cascade(label='Colors', menu=colormenu)
        # colormenu.add_command(label='Brush Color', command=self.change_fg)
        # colormenu.add_command(label='Background Color', command=self.change_bg)
        optionmenu = Menu(menu)
        menu.add_cascade(label='Options', menu=optionmenu)
        optionmenu.add_command(label='Clear canvas', command=self.clear)
        optionmenu.add_command(label='Exit', command=self.master.destroy)
        optionmenu.add_command(label="Save coloured map", command=lambda: self.save_label_map(mode='RGB'))
        optionmenu.add_command(label="Save grey map", command=lambda: self.save_label_map(mode='GRAY'))
        optionmenu.add_command(label="Save synthesized image", command=lambda: self.save_last_synth_img())
        optionmenu.add_command(label="Save SAPDE label", command=lambda: self.save_last_SPADE_label())
        optionmenu.add_command(label="Generate picture", command=self.generate_picture)
        optionmenu.add_command(label="Load label map", command=lambda: self.load_label_map())
        optionmenu.add_command(label="Load example", command=lambda: self.load_example())
        optionmenu.add_command(label="Detect objects", command=lambda: self.detect_objects())


if __name__ == '__main__':
    root = Tk()
    main(root)
    root.title('Application')
    root.mainloop()

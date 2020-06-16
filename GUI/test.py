from tkinter import *
from tkinter import ttk, colorchooser
from PIL import Image
import cv2
import os
import io
import numpy as np


class main:
    def __init__(self, master):
        self.master = master
        self.color_fg = '#22aaff'
        self.color_bg = '#22aaff'
        self.old_x = None
        self.old_y = None
        self.penwidth = 5
        self.drawWidgets()
        self.c.bind('<B1-Motion>', self.paint)  # drwaing the line
        self.c.bind('<ButtonRelease-1>', self.reset)
        self.rgb_grey_id_dict = {
            {}
        }

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

    #def change_fg(self, class_):  # changing the pen color
     #   self.color_fg = colorchooser.askcolor(color=self.color_fg)[1]

    def tree_fg(self):  # changing the pen color
        self.color_fg = "#00ff00"

    def person_fg(self):  # changing the pen color
        self.color_fg = "#ff0000"

    def bridge_fg(self):  # changing the pen color
        self.color_fg = "#0000ff"

    def sky_other_fg(self):  # changing the pen color
        self.color_fg = "#22aaff"

    def bicycle_fg(self):
        self.color_fg = "#606f74"

    def car_fg(self):
        self.color_fg = "#180d2f"

    def motorcycle_fg(self):
        self.color_fg = "#626313"

    def bus_fg(self):
        self.color_fg = "#ebdbe4"

    def train_fg(self):
        self.color_fg = "#ed261d"

    def truck_fg(self):
        self.color_fg = "#0e170d"

    def boat_fg(self):
        self.color_fg = "#50426c"

    def traffic_light(self):
        self.color_fg = "#e22ec5"

    def fire_hydrant_fg(self):
        self.color_fg = "#d304f5"

    def street_sign(self):
        self.color_fg = "#eaae4d"

    def stop_sign_fg(self):
        self.color_fg = "#d304f5"

    def parking_meter_fg(self):
        self.color_fg = "#ddd144"

    def bench_fg(self):
        self.color_fg = "#626313"

    def bird_fg(self):
        self.color_fg = "#f8c8ac"

    def cat_fg(self):
        self.color_fg = "#908720"

    def dog_fg(self):
        self.color_fg = "#e76555"

    def horse_fg(self):
        self.color_fg = "#c6b5b4"

    def sheep_fg(self):
        self.color_fg = "#dee9b7"

    def cow_fg(self):
        self.color_fg = "#7f1cd1"

    def elephant_fg(self):
        self.color_fg = "#a0b399"

    def bear_fg(self):
        self.color_fg = "#957f10"

    def zebra_fg(self):
        self.color_fg = "#f9e5d6"

    def giraffe_fg(self):
        self.color_fg = "#b74936"

    def hat_fg(self):
        self.color_fg = "#b805dc"

    def backpack_fg(self):
        self.color_fg = "#683d38"

    def umbrella_fg(self):
        self.color_fg = "#0b40b6"

    def shoe_fg(self):
        self.color_fg = "#240a48"

    def eye_glasses_fg(self):
        self.color_fg = "#a0b399"

    def handbag_fg(self):
        self.color_fg = "#b46355"

    def tie_fg(self):
        self.color_fg = "#c81014"

    def suitcase_fg(self):
        self.color_fg = "#7f1cd1"

    def frisbee_fg(self):
        self.color_fg = "#0fd0b9"

    def skis_fg(self):
        self.color_fg = "#f55cd6"

    def snowboard_fg(self):
        self.color_fg = "#633345"

    def sports_ball_fg(self):
        self.color_fg = "#ceda3d"

    def kite_fg(self):
        self.color_fg = "#8977fa"

    def baseball_bat_fg(self):
        self.color_fg = "#eff9a3"

    def baseball_glove_fg(self):
        self.color_fg = "#b46355"

    def skateboard_fg(self):
        self.color_fg = "#023263"

    def surfboard_fg(self):
        self.color_fg = "#f19999"

    def tennis_racket_fg(self):
        self.color_fg = "#9493eb"

    def bottle_fg(self):
        self.color_fg = "#918a91"

    def plate_fg(self):
        self.color_fg = "#f19999"

    def wine_glass_fg(self):
        self.color_fg = "#d96d40"

    def cup_fg(self):
        self.color_fg = "#14e246"

    def fork_fg(self):
        self.color_fg = "#75e26c"

    def knife_fg(self):
        self.color_fg = "#8db834"

    def spoon_fg(self):
        self.color_fg = "#577e2b"

    def bowl_fg(self):
        self.color_fg = "#300e8a"

    def banana_fg(self):
        self.color_fg = "#ddb011"

    def apple_fg(self):
        self.color_fg = "#de0304"

    def sandwich_fg(self):
        self.color_fg = "#c0a92a"

    def orange_fg(self):
        self.color_fg = "#e05f27"

    def broccoli_fg(self):
        self.color_fg = "#086953"

    def carrot_fg(self):
        self.color_fg = "#de0304"

    def hot_dog_fg(self):
        self.color_fg = "#f7d20e"

    def pizza_fg(self):
        self.color_fg = "#8a364c"

    def donut_fg(self):
        self.color_fg = "#8c6e28"

    def cake_fg(self):
        self.color_fg = "#e46085"

    def chair_fg(self):
        self.color_fg = "#6c2818"

    def couch_fg(self):
        self.color_fg = "#f2d3e9"

    def potted_plant_fg(self):
        self.color_fg = "#62d50f"

    def bed_fg(self):
        self.color_fg = "#f2d3e9"

    def mirror_stuff_fg(self):
        self.color_fg = "#87a5cd"

    def dining_table_fg(self):
        self.color_fg = "#d7c391"

    def window_fg(self):
        self.color_fg = "#665988"

    def desk_fg(self):
        self.color_fg = "#aa4967"

    def toilet_fg(self):
        self.color_fg = "#7b31ed"

    def door_fg(self):
        self.color_fg = "#fe4446"

    def tv_fg(self):
        self.color_fg = "#0a365a"

    def laptop_fg(self):
        self.color_fg = "#bcdfd7"

    def mouse_fg(self):
        self.color_fg = "#4749a4"

    def remote_fg(self):
        self.color_fg = "#dc308c"

    def keyboard_fg(self):
        self.color_fg = "#d28535"

    def cell_phone_fg(self):
        self.color_fg = "#eb69b6"

    def microwave_fg(self):
        self.color_fg = "#494171"

    def oven_fg(self):
        self.color_fg = "#174f46"

    def toaster_fg(self):
        self.color_fg = "#1c575f"

    def sink_fg(self):
        self.color_fg = "#ddf289"

    def refrigerator_fg(self):
        self.color_fg = "#9d68bd"

    def blender_fg(self):
        self.color_fg = "#9f714a"

    def book_fg(self):
        self.color_fg = "#715fcc"

    def clock_fg(self):
        self.color_fg = "#6c9617"

    def vase_fg(self):
        self.color_fg = "#f78adb"

    def scissors_fg(self):
        self.color_fg = "#ee3bb3"

    def teddy_bear_fg(self):
        self.color_fg = "#9f714a"

    def hair_drier_fg(self):
        self.color_fg = "#f014a1"

    def toothbrush_fg(self):
        self.color_fg = "#74dbfe"

    def hair_brush_fg(self):  # last class of Thing
        self.color_fg = "#dd7cc3"

    def banner_fg(self):  # first class of Stuff
        self.color_fg = "#6c9617"

    def blanket_fg(self):
        self.color_fg = "#bdc8a2"

    def branch_fg(self):
        self.color_fg = "#584e26"

    def building_other_fg(self):
        self.color_fg = "#584f2f"

    def bush_fg(self):
        self.color_fg = "#0e8809"

    def cabinet_fg(self):
        self.color_fg = "#103dae"

    def cage_fg(self):
        self.color_fg = "#c09003"

    def cardboard_fg(self):
        self.color_fg = "#1d3c07"

    def carpet_fg(self):
        self.color_fg = "#d55509"

    def ceiling_other_fg(self):
        self.color_fg = "#d0f790"

    def ceiling_tile_fg(self):
        self.color_fg = "#4fe78c"

    def cloth_fg(self):
        self.color_fg = "#ed5f29"

    def clothes_fg(self):
        self.color_fg = "#a55462"

    def clouds_fg(self):
        self.color_fg = "#74dbfe"

    def counter_fg(self):
        self.color_fg = "#025623"

    def cupboard_fg(self):
        self.color_fg = "#76fa08"

    def curtain_fg(self):
        self.color_fg = "#aa4967"

    def desk_stuff_fg(self):
        self.color_fg = "#e4db52"

    def dirt_fg(self):
        self.color_fg = "#7fae9e"

    def door_stuff_fg(self):
        self.color_fg = "#7b12b7"

    def fence_fg(self):
        self.color_fg = "#b7bf62"

    def floor_marble_fg(self):
        self.color_fg = "#0c2a5f"

    def floor_other_fg(self):
        self.color_fg = "#657b9c"

    def floor_stone_fg(self):
        self.color_fg = "#7aa3ea"

    def floor_tile_fg(self):
        self.color_fg = "#b2fbc7"

    def floor_wood_fg(self):
        self.color_fg = "#7fa54e"

    def flower_fg(self):
        self.color_fg = "#fd727c"

    def fog_fg(self):
        self.color_fg = "#7fae9e"

    def food_other_fg(self):
        self.color_fg = "#942bca"

    def fruit_fg(self):
        self.color_fg = "#c7510e"

    def furniture_other_fg(self):
        self.color_fg = "#5a060b"

    def grass_fg(self):
        self.color_fg = "#38700c"

    def gravel_fg(self):
        self.color_fg = "#0b3d6e"

    def ground_other_fg(self):
        self.color_fg = "#ad741a"

    def hill_fg(self):
        self.color_fg = "#bcef69"

    def house_fg(self):
        self.color_fg = "#e95974"

    def leaves_fg(self):
        self.color_fg = "#25d626"

    def light_fg(self):
        self.color_fg = "#f5f222"

    def mat_fg(self):
        self.color_fg = "#0723a5"

    def metal_fg(self):
        self.color_fg = "#657b9c"

    def moss_fg(self):
        self.color_fg = "#25d626"

    def mountain_fg(self):
        self.color_fg = "76b7b2"

    def mud_fg(self):
        self.color_fg = "#4a351a"

    def napkin_fg(self):
        self.color_fg = "#837acb"

    def net_fg(self):
        self.color_fg = "#c7510e"

    def paper_fg(self):
        self.color_fg = "#b6ccbe"

    def pavement_fg(self):
        self.color_fg = "#a7abc2"

    def pillow_fg(self):
        self.color_fg = "#ac6dd0"

    def plant_other_fg(self):
        self.color_fg = "#149688"

    def plastic_fg(self):
        self.color_fg = "#6c5351"

    def platform_fg(self):
        self.color_fg = "#fe46f6"

    def playingfield_fg(self):
        self.color_fg = "#9e0b30"

    def railing_fg(self):
        self.color_fg = "#4d0d98"

    def railroad_fg(self):
        self.color_fg = "#7dd8d9"

    def river_fg(self):
        self.color_fg = "#7653fd"

    def road_fg(self):
        self.color_fg = "#4d0c3d"

    def rock_fg(self):
        self.color_fg = "#81656d"

    def roof_fg(self):
        self.color_fg = "#0e471e"

    def rug_fg(self):
        self.color_fg = "#fe46f6"

    def salad_fg(self):
        self.color_fg = "#8b5e61"

    def sand_fg(self):
        self.color_fg = "#dbc386"

    def sea_fg(self):
        self.color_fg = "#10ebf2"

    def shelf_fg(self):
        self.color_fg = "#65731f"

    def skyscraper_fg(self):
        self.color_fg = "#9d1670"

    def snow_fg(self):
        self.color_fg = "#ecd9fe"

    def solid_other_fg(self):
        self.color_fg = "#05b8dc"

    def stairs_fg(self):
        self.color_fg = "#8f0905"

    def stone_fg(self):
        self.color_fg = "#a7abc2"

    def straw_fg(self):
        self.color_fg = "#4cad63"

    def structural_other_fg(self):
        self.color_fg = "#b6ccbe"

    def table_fg(self):
        self.color_fg = "#9d8356"

    def tent_fg(self):
        self.color_fg = "#b519e7"

    def textile_other_fg(self):
        self.color_fg = "#f91c1c"

    def towel_fg(self):
        self.color_fg = "#4625f5"

    def vegetable_fg(self):
        self.color_fg = "#b96fd4"

    def wall_concrete_fg(self):
        self.color_fg = "#3a364c"

    def wall_brick_fg(self):
        self.color_fg = "#151f09"

    def wall_other_fg(self):
        self.color_fg = "#ab11b5"

    def wall_wood_fg(self):
        self.color_fg = "#978f48"

    def wall_panel_fg(self):
        self.color_fg = "#65ef40"

    def wall_stone_fg(self):
        self.color_fg = "#8fada7"

    def wall_tile_fg(self):
        self.color_fg = "#3b2b64"

    def water_other_fg(self):
        self.color_fg = "#51d2ec"

    def waterdrops_fg(self):
        self.color_fg = "#05b8dc"

    def window_blind_fg(self):
        self.color_fg = "#d3476e"

    def window_other_fg(self):
        self.color_fg = "#dfe39d"

    def wood_fg(self):
        self.color_fg = "#743975"

    def airplane_fg(self):
        self.color_fg = "#fd6210"

    def mirror_fg(self):
        self.color_fg = "#55ebd2"

    # def change_bg(self):  # changing the background color canvas
      #   self.color_bg = colorchooser.askcolor(color=self.color_bg)[1]
      #   self.c['bg'] = self.color_bg

    def save_as_rgb_image(self):
        canvas = self.c.postscript(colormode="color")
        img = np.asarray(Image.open(io.BytesIO(canvas.encode("utf-8"))))
        cv2.imwrite("/home/bernihoh/Bachelor/GUI-test-pictures/a.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def save_as_grey_image(self):
        canvas = self.c.postscript(colormode="color")
        img_a = np.asarray(Image.open(io.BytesIO(canvas.encode("utf-8"))))
        # imga.save("/home/bernihoh/Bachelor/GUI-test-pictures/a_grey" + ".png", "png")
        img = cv2.cvtColor(img_a, cv2.COLOR_RGB2BGR)
        (height, width) = img.size
        for i in range(height):
            for j in range(width):
                if img[j][i] is [255, 0, 0]:
                    img[j][i] = 10
                elif img[j][i] is [0, 255, 0]:
                    img[j][i] = 50
                elif img[j][i] is [0, 0, 255]:
                    img[j][i] = 100
        cv2.imwrite("/home/bernihoh/Bachelor/GUI-test-pictures/a_grey" + ".png", img)

    def drawWidgets(self):
        self.controls = Frame(self.master, padx=5, pady=5)
        Label(self.controls, text='Pen Width:', font=('arial 18')).grid(row=0, column=0)
        self.slider = ttk.Scale(self.controls, from_=1, to=150, command=self.changeW, orient=VERTICAL)
        self.slider.set(self.penwidth)
        self.slider.grid(row=0, column=1, ipadx=30)
        self.controls.pack(side=LEFT)

        self.c = Canvas(self.master, width=500, height=400, bg=self.color_bg)
        self.c.pack(fill=BOTH, expand=True)

        menu = Menu(self.master)
        self.master.config(menu=menu)
        filemenu = Menu(menu)
        colormenu = Menu(menu)
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
        ## class_chooser = Menu(menu)
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
        vehicle_roads.add_command(label="traffic light", command=lambda: self.traffic_light())
        vehicle_roads.add_command(label="fire hydrant", command=lambda: self.fire_hydrant_fg())
        vehicle_roads.add_command(label="street sign", command=lambda: self.street_sign())
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
        optionmenu.add_command(label='Clear Canvas', command=self.clear)
        optionmenu.add_command(label='Exit', command=self.master.destroy)
        optionmenu.add_command(label="Save as rgb image", command=self.save_as_rgb_image)
        optionmenu.add_command(label="Save as grey image", command=self.save_as_grey_image)

if __name__ == '__main__':
    root = Tk()
    main(root)
    root.title('Application')
    root.mainloop()

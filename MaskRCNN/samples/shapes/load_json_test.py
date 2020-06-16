import json
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import pprint

segmentation_id_map_stuff = []
segmentation_id_map_thing = []
image_id_segmentation_stuff_thing_map = []
proz_array = [None]*5
append_all_proz_array = []
proz_array[0] = image_id_segmentation_stuff_thing_map
proz_array[1] = image_id_segmentation_stuff_thing_map
json_file_dir_thing = "/home/bernihoh/Bachelor/cocostuffapi/PythonAPI/pycocotools/annotations/instances_train2017.json"
annotations_file_thing = json.load(open(json_file_dir_thing))
annotations_thing = annotations_file_thing["annotations"]
image_ids_thing = [b["image_id"] for b in annotations_thing]
image_ids_thing = sorted(image_ids_thing)
annotations_thing_sorted = []
json_file_dir_stuff = "/home/bernihoh/Bachelor/cocostuffapi/PythonAPI/pycocotools/stuff_train2017.json"
annotations_file_stuff = json.load(open(json_file_dir_stuff))
annotations_stuff = annotations_file_stuff["annotations"]
annotations_stuff_sorted = []
print(image_ids_thing)
image_ids_thing_condensed = []
old_image_id = -1
print("condensing")
for image_id in image_ids_thing:
    if old_image_id < image_id:
        old_image_id = image_id
        image_ids_thing_condensed.append(image_id)
print("End condensing")
print(len(image_ids_thing))
print(len(image_ids_thing_condensed))
print(len(annotations_stuff))
i = 0
print("sorting")
for image_id in image_ids_thing_condensed:
    # i += 1
    # print(i)
    annotations_for_picture_thing = [b for b in annotations_thing if b["image_id"] == image_id]
    annotations_for_picture_stuff = [c for c in annotations_stuff if c["image_id"] == image_id]
    for d in annotations_for_picture_thing:
        annotations_thing_sorted.append(d)
    for e in annotations_for_picture_stuff:
        annotations_stuff_sorted.append(e)
print("End sorting")
annotations_thing = annotations_thing_sorted
annotations_stuff = annotations_stuff_sorted
print(len(annotations_thing))
print(len(annotations_stuff))



# annotations_for_picture_thing = [a["segmentation"] for a in annotations_thing if a["image_id"] == image_ids_thing[0]]
# image_id_for_picture_thing = [a["image_id"] for a in annotations_thing]
# print(annotations_for_picture_thing)




# for a in range(len(image_ids_thing)):
for j in range(2):
    print(image_ids_thing[j])
    annotations_for_picture_thing = [b for b in annotations_thing if b["image_id"] == image_ids_thing[j]]
    print(len(annotations_for_picture_thing))
    for c in annotations_for_picture_thing:
        item_of_segmentation_id_map_thing = {"segment": c["segmentation"], "category_id": c["category_id"]}
        segmentation_id_map_thing.append(item_of_segmentation_id_map_thing)
    annotations_for_picture_stuff = [c for c in annotations_stuff if c["image_id"] == image_ids_thing[j]]
    for d in annotations_for_picture_stuff:
        item_of_segmentation_id_map_stuff = {"segment": d["bbox"], "category_id": d["category_id"]}
        segmentation_id_map_stuff.append(item_of_segmentation_id_map_stuff)
    item_of_image_id_segmentation_stuff_thing_map = {"image_id": image_ids_thing[j],
                                                     "segmentation_id_map_thing": segmentation_id_map_thing,
                                                     "segmentation_id_map_stuff": segmentation_id_map_stuff}
    append_all_proz_array.append(item_of_image_id_segmentation_stuff_thing_map)

    # Man muss die folgenden 2 Listen löschen:
    segmentation_id_map_thing = []
    segmentation_id_map_stuff = []

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(append_all_proz_array)

# annotations_for_picture_stuff = [c for c in annotations_stuff if c["image_id"] == image_ids_thing[0]]
# bbox_annForPic_stuff = [d["bbox"] for d in annotations_for_picture_stuff]
# print(bbox_annForPic_stuff)
# for i in range(len(proz_array)):
    # append_all_proz_array.append(proz_array[i])
  #


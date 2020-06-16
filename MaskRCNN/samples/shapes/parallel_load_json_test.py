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
import sys
import pprint
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from threading import Barrier

b = Barrier(parties=mp.cpu_count())


def parallel_func_old(processor_number, processor_count, image_ids_thing,
                      annotations_thing, start_point, annotations_stuff,  iterator_max):
    #
    # annotations_thing = json.loads(annotations_thing)
    print(processor_number, "Start")
    segmentation_id_map_stuff = []
    segmentation_id_map_thing = []
    item_of_image_id_segmentation_stuff_thing_map = {}
    # iterator_max = int(len_img_ids_thing / processor_count)
    result_array = []

    for i in range(iterator_max):
        asked_image = start_point + processor_number + i * processor_count

        annotations_for_picture_thing = [b for b in annotations_thing
                                         if b["image_id"] == image_ids_thing[asked_image]]
        for c in annotations_for_picture_thing:
            item_of_segmentation_id_map_thing = {"segment": c["segmentation"], "category_id": c["category_id"]}
            segmentation_id_map_thing.append(item_of_segmentation_id_map_thing)
        annotations_for_picture_stuff = [c for c in annotations_stuff if c["image_id"] == image_ids_thing[asked_image]]

        for d in annotations_for_picture_stuff:
            item_of_segmentation_id_map_stuff = {"segment": d["bbox"], "category_id": d["category_id"]}
            segmentation_id_map_stuff.append(item_of_segmentation_id_map_stuff)
        item_of_image_id_segmentation_stuff_thing_map = {"image_id": image_ids_thing[asked_image],
                                                         "segmentation_id_map_thing": segmentation_id_map_thing,
                                                         "segmentation_id_map_stuff": segmentation_id_map_stuff}

        result_array.append(item_of_image_id_segmentation_stuff_thing_map)
        # print(processor_number)
        # Man muss die folgenden 2 Listen löschen:
        segmentation_id_map_thing = []
        segmentation_id_map_stuff = []

    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(result_array)
    print(processor_number, "End")
    # b.wait()
    # final_result_array[processor_number] = result_array
    # print(result_array)
    return result_array


def thread_awakener():
    # for RAM Saving
    PICTURE_CAP = 10000  # This is ok for 64 GB of RAM
    # for multiprocessing
    processor_count = mp.cpu_count()


    json_file_dir_thing = "/home/bernihoh/Bachelor/cocostuffapi/PythonAPI/pycocotools/annotations/instances_train2017.json"
    annotations_file_thing = json.load(open(json_file_dir_thing))
    annotations_thing = annotations_file_thing["annotations"]
    json_file_dir_stuff = "/home/bernihoh/Bachelor/cocostuffapi/PythonAPI/pycocotools/stuff_train2017.json"
    annotations_file_stuff = json.load(open(json_file_dir_stuff))
    annotations_stuff = annotations_file_stuff["annotations"]
    annotations_thing_stuff_sorted = []
    image_ids_thing = [b["image_id"] for b in annotations_thing]
    image_ids_thing_condensed = []
    old_image_id = -1
    for image_id in image_ids_thing:
        if old_image_id < image_id:
            old_image_id = image_id
            image_ids_thing_condensed.append(image_id)
    # evntl. eine Tabelle mit den Grenzen!!!!!!!!!
    for image_id in image_ids_thing_condensed:
        annotations_for_picture_thing = [b for b in annotations_thing if b["image_id"] == image_id]
        annotations_for_picture_stuff = [c for c in annotations_stuff if c["image_id"] == image_id]
        for d in annotations_for_picture_thing:
            annotations_thing_stuff_sorted.append(d)
        for e in annotations_for_picture_stuff:
            annotations_thing_stuff_sorted.append(e)
    len_annotations_thing_stuff_sorted = len(annotations_thing_stuff_sorted)
    print(len_annotations_thing_stuff_sorted)
    len_divided_annotations_thing_stuff_array = int(len_annotations_thing_stuff_sorted / PICTURE_CAP) + 1
    divided_annotations_thing_stuff_array = [None] * len_divided_annotations_thing_stuff_array

    if len_annotations_thing_stuff_sorted > PICTURE_CAP:
        cap_counter = 0
        divided_annotations_thing_stuff_array_position = 0
        picture_array_thing_stuff = []
        for image_id in image_ids_thing_condensed:
            annotations_for_picture_thing_stuff = [a for a in annotations_thing_stuff_sorted
                                                   if a["image_id"] == image_id]
            cap_counter += len(annotations_for_picture_thing_stuff)
            if cap_counter <= PICTURE_CAP:
                for b in annotations_for_picture_thing_stuff:
                    picture_array_thing_stuff.append(b)
            else:
                divided_annotations_thing_stuff_array[divided_annotations_thing_stuff_array_position] = \
                    picture_array_thing_stuff
                divided_annotations_thing_stuff_array_position += 1
                picture_array_thing_stuff = []
                for c in annotations_for_picture_thing_stuff:
                    picture_array_thing_stuff.append(c)
                cap_counter = len(annotations_for_picture_thing_stuff)

    print("ready with dividing")
    # print(image_ids_thing)
    len_img_ids_thing = len(image_ids_thing)
    print(len_img_ids_thing)

    for j in range(len_divided_annotations_thing_stuff_array):
        pool = mp.Pool(processor_count)
        print("run", j)
        iterator_max = int(len(divided_annotations_thing_stuff_array[j]) / processor_count)
        start_point = j * PICTURE_CAP
        print("start_point=", start_point)
        # print("array:", divided_annotations_thing_array[j])

        print("Size:", sys.getsizeof(divided_annotations_thing_array[j]))

        results_objects = [pool.apply_async(parallel_func, args=(i, processor_count, image_ids_thing,
                                                                 divided_annotations_thing_array[j],
                                                                 start_point, annotations_stuff, iterator_max))
                           for i in range(processor_count)]

        pool.close()
        pool.join()
        results = [r.get()[1] for r in results_objects]
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(results)


if __name__ == '__main__':
    thread_awakener()

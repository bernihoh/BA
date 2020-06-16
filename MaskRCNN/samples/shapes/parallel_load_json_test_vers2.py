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

PICTURE_CAP = 1000  # This is ok for 64 GB of RAM


def parallel_sort(proz, divided_annotations_array, image_ids_condensed_sorted):
    """
    This function sorts "divided_annotations_array" and returns "divided_annotations_sorted"
    with the "border_table" of the "image_ids" in  "divided_annotations_sorted".
    """
    # needed vars
    len_divided_annotations_array = len(divided_annotations_array)

    # needed main arrays
    divided_annotations_sorted = []
    divided_annotations_chunk_sorted = []
    chunk_sorted = []
    divided_annotations_array_chunks_sorted = []
    border_table = []
    divided_annotations_array_chunks = [None] * (int(len_divided_annotations_array / PICTURE_CAP) + 1)
    if proz == 31: print((int(len_divided_annotations_array / PICTURE_CAP) + 1))
    border_table_chunk = []
    border_table_chunks_array = []

    # needed vars
    len_divided_annotations_array_chunks = len(divided_annotations_array_chunks)

    # dividing "divided_annotations_array" further into chunks of size PICTURE_CAP
    if proz == 31: print("dividing divided_annotations_array further into chunks of size PICTURE_CAP")
    count = 0
    for i in range(len_divided_annotations_array_chunks):
        start = i * PICTURE_CAP
        stop = start + PICTURE_CAP
        if stop > len_divided_annotations_array:
            stop = len_divided_annotations_array
        divided_annotations_array_chunks[i] = divided_annotations_array[start:stop]

    # Sorting every chunk
    if proz == 31: print("Sorting every chunk")
    count = 0
    for chunk in divided_annotations_array_chunks:
        if proz == 31:
            count += 1
            print(count)
        end_image_id_pos = 0
        for image_id in image_ids_condensed_sorted:
            start_image_id_pos = end_image_id_pos
            annotations_for_picture = [a for a in chunk if a["image_id"] == image_id]
            len_annotations_for_picture = len(annotations_for_picture)
            end_image_id_pos += len_annotations_for_picture
            chunk_sorted += annotations_for_picture
            border_table_chunk_entry = {"image_id": image_id, "start_image_id_pos": start_image_id_pos,
                                        "end_image_id_pos": end_image_id_pos}
            #    border_table_chunk_entry = {"image_id": image_id, "start_image_id_pos": -1, "end_image_id_pos": -1}
            # if proz == 31: print(border_table_chunk_entry)
            border_table_chunk.append(border_table_chunk_entry)
        divided_annotations_array_chunks_sorted.append(chunk_sorted)
        chunk_sorted = []
        border_table_chunks_array.append(border_table_chunk)
        border_table_chunk = []


    # Zipping every chunk to one sorted array
    if proz == 31: print("Zipping every chunk to one sorted array")
    end_of_image_id = 0
    for j in range(len(image_ids_condensed_sorted)):
        image_id_piece_of_all_chunks = []
        start_of_image_id = end_of_image_id
        count = 0
        for b_t in border_table_chunks_array:
            start_image_id_pos_in_chunk = b_t[j]["start_image_id_pos"]
            end_image_id_pos_in_chunk = b_t[j]["end_image_id_pos"]
            assert b_t[j]["image_id"] == image_ids_condensed_sorted[j]
            # if proz == 31:
              #   print(start_image_id_pos_in_chunk, end_image_id_pos_in_chunk)
            # if end_image_id_pos_in_chunk > -1:
            length = (end_image_id_pos_in_chunk - start_image_id_pos_in_chunk)
            end_of_image_id += length
            image_id_piece_array_in_chunk = divided_annotations_array_chunks_sorted[count][start_image_id_pos_in_chunk:
                                                                                           end_image_id_pos_in_chunk]

            image_id_piece_of_all_chunks += image_id_piece_array_in_chunk
            image_id_piece_array_in_chunk = []
            count += 1
        border_table_entry = {"image_id": image_ids_condensed_sorted[j], "start_image_id_pos": start_of_image_id,
                              "end_image_id_pos": end_of_image_id}
        border_table.append(border_table_entry)

        divided_annotations_sorted += image_id_piece_of_all_chunks

    return_element = {"divided_annotations_sorted": divided_annotations_sorted, "border_table": border_table}

    return return_element


def parallel_return_concat_thing_stuff(proz, annotations_stuff, border_table_stuff,
                                       annotations_thing, border_table_thing):
    """
    Returns the concatenated array "concat_annotations" of "annotations_stuff" and "annotations_thing".
    Returns the concatenated array "concat_border_table" of "border_table_stuff" and "border_table_thing".
    """
    concat_annotations = []
    concat_border_table = []
    start_image_id = 0
    end_image_id = 0
    length = 0
    length_counter = 0
    if proz == 31: print("Concatenating: annotations_stuff with annotations_thing and border_table_stuff with "
                         "border_table_thing")
    for i in range(len(border_table_stuff)):
        if proz == 31: print(i)
        length = 0
        concat_annotations_piece = []
        concat_border_table_entry = []
        annotations_stuff_piece = []
        asked_image_id = border_table_stuff[i]["image_id"]
        start_of_annotations_stuff_piece = border_table_stuff[i]["start_image_id_pos"]
        end_of_annotations_stuff_piece = border_table_stuff[i]["end_image_id_pos"]
        if end_of_annotations_stuff_piece > -1:
            length += (end_of_annotations_stuff_piece - start_of_annotations_stuff_piece)
        else:
            length += 0
        if end_of_annotations_stuff_piece > -1:
            annotations_stuff_piece = annotations_stuff[start_of_annotations_stuff_piece:
                                                        end_of_annotations_stuff_piece]
        annotations_thing_piece = []
        start_of_annotations_thing_piece = border_table_thing[i]["start_image_id_pos"]
        end_of_annotations_thing_piece = border_table_thing[i]["end_image_id_pos"]
        # if proz == 31:
            # print(start_of_annotations_thing_piece, end_of_annotations_thing_piece)
        if end_of_annotations_thing_piece > -1:
            length += (end_of_annotations_thing_piece - start_of_annotations_thing_piece)
        else:
            length += 0
        if end_of_annotations_thing_piece > -1:
            annotations_thing_piece = annotations_thing[start_of_annotations_thing_piece:
                                                        end_of_annotations_thing_piece]
        end_image_id = start_image_id + length
        length_counter += length
        concat_annotations_piece = annotations_stuff_piece + annotations_thing_piece
        concat_border_table_entry = [{"image_id": asked_image_id, "start_image_id_pos": start_image_id,
                                     "end_image_id_pos_penis": end_image_id}]

        start_image_id = end_image_id
        concat_annotations = concat_annotations + concat_annotations_piece
        concat_border_table += concat_border_table_entry

    return_element = {"concat_annotations": concat_annotations, "concat_border_table": concat_border_table, "length_counter": length_counter}
    return return_element


def master(part):
    assert part in ["val", "train"]
    # needed Vars
    processor_count = mp.cpu_count()

    # needed main arrays
    result_array_of_parallelization = []
    image_ids_condensed_sorted = []

    # loading annotations from jsons
    print("Start: loading annotations from jsons")
    json_file_dir_thing = "/home/bernihoh/Bachelor/cocostuffapi/PythonAPI/pycocotools/annotations/instances_"+part+"2017.json"
    annotations_file_thing = json.load(open(json_file_dir_thing))
    annotations_thing = annotations_file_thing["annotations"]
    json_file_dir_stuff = "/home/bernihoh/Bachelor/cocostuffapi/PythonAPI/pycocotools/stuff_"+part+"2017.json"
    annotations_file_stuff = json.load(open(json_file_dir_stuff))
    annotations_stuff = annotations_file_stuff["annotations"]
    print("End: loading annotations from jsons")

    # Sorting and condensing image_ids
    print("Start: Sorting and condensing image_ids")
    image_ids_thing = [b["image_id"] for b in annotations_thing]
    image_ids_stuff = [b["image_id"] for b in annotations_stuff]
    image_ids_thing = sorted(image_ids_thing)
    print("Length_thing:", len(image_ids_thing))
    print("Length_stuff:", len(image_ids_stuff))
    old_image_id = -1
    for image_id in image_ids_thing:
        if old_image_id < image_id:
            old_image_id = image_id
            image_ids_condensed_sorted.append(image_id)
    print("Length condensed:", len(image_ids_condensed_sorted))
    print("End: Sorting and condensing image_ids")

    # Dividing "annotations_thing" into "processor_count" many chunks of equal size
    print("Start: Dividing annotations_thing")
    len_annotations_thing = len(annotations_thing)
    chunk_size_float = len_annotations_thing / processor_count
    chunk_size = int(len_annotations_thing / processor_count)
    if chunk_size_float > chunk_size:
        chunk_size = chunk_size + 1
    start = 0
    stop = chunk_size
    divided_annotations_thing_array = [None] * processor_count
    for i in range(processor_count):
        if stop > len_annotations_thing:
            stop = len_annotations_thing
        divided_annotations_thing_array[i] = annotations_thing[start:stop]
        start += chunk_size
        stop += chunk_size
    print("End: Dividing annotations_thing")

    # Dividing "annotations_stuff" into "processor_count" many chunks of equal size
    print("Start: Dividing annotations_stuff")
    len_annotations_stuff = len(annotations_stuff)
    chunk_size_float = len_annotations_stuff / processor_count
    chunk_size = int(len_annotations_stuff / processor_count)
    if chunk_size_float > chunk_size:
        chunk_size = chunk_size + 1
    start = 0
    stop = chunk_size
    divided_annotations_stuff_array = [None] * processor_count
    for i in range(processor_count):
        if stop > len_annotations_thing:
            stop = len_annotations_thing
        divided_annotations_stuff_array[i] = annotations_stuff[start:stop - 1]
        start += chunk_size
        stop += chunk_size
    print("End: Dividing annotations_stuff")

    # Calling functions "parallel_sort" and "parallel_return_pictures_of_image_id" to sort the "annotations_thing" and
    # "annotations_stuff" arrays
    print("Start: parallel_sort")
    pool = mp.Pool(processor_count)
    results_objects_thing_sorted = [pool.apply_async(parallel_sort, args=(i, divided_annotations_thing_array[i],
                                                                          image_ids_condensed_sorted))
                                    for i in range(processor_count)]
    pool.close()
    pool.join()
    results_thing_sorted = [r.get() for r in results_objects_thing_sorted]
    pool = mp.Pool(processor_count)
    results_objects_stuff_sorted = [pool.apply_async(parallel_sort, args=(i, divided_annotations_stuff_array[i],
                                                                          image_ids_condensed_sorted))
                                    for i in range(processor_count)]
    pool.close()
    pool.join()
    results_stuff_sorted = [r.get() for r in results_objects_stuff_sorted]

    print("End: parallel_sort")

    # Zipping every process result of thing and stuff to one sorted array
    print("Start: Zipping every process result of thing and stuff to one sorted array")
    pool = mp.Pool(processor_count)
    results_objects_thing_stuff_sorted_concat = [pool.apply_async(parallel_return_concat_thing_stuff,
                                                                  args=(i,
                                                                        results_stuff_sorted[i]["divided_annotations_sorted"],
                                                                        results_stuff_sorted[i]["border_table"],
                                                                        results_thing_sorted[i]["divided_annotations_sorted"],
                                                                        results_thing_sorted[i]["border_table"]))
                                                 for i in range(processor_count)]

    results_thing_stuff_sorted_concat = [r.get() for r in results_objects_thing_stuff_sorted_concat]
    counter = 0
    final_border_table = []
    final_annotations = []
    end_pos = 0
    for i in range(len(image_ids_condensed_sorted)):
        start_pos = end_pos
        total_image_id_piece = []
        total_length = 0
        for concat in results_thing_stuff_sorted_concat:

            image_id_piece_in_concat = []
            start_of_image_id_in_concat = concat["concat_border_table"][i]["start_image_id_pos"]
            end_of_image_id_in_concat = concat["concat_border_table"][i]["end_image_id_pos_penis"]
            assert image_ids_condensed_sorted[i] == concat["concat_border_table"][i]["image_id"]
            total_length += (end_of_image_id_in_concat - start_of_image_id_in_concat)
            image_id_piece_in_concat = concat["concat_annotations"][start_of_image_id_in_concat:
                                                                    end_of_image_id_in_concat ]
            total_image_id_piece += image_id_piece_in_concat
        end_pos += total_length
        final_border_table_entry = [{"image_id": image_ids_condensed_sorted[i], "start_image_id_pos": start_pos,
                                     "end_image_id_pos": end_pos}]
        assert start_pos <= end_pos
        final_border_table.append(final_border_table_entry)
        final_annotations += total_image_id_piece
    pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(final_annotations)
    print(len(final_border_table), len(final_annotations))
    print("End: Zipping every process result of thing and stuff to one sorted array")
    with open("/home/bernihoh/Bachelor/cocostuffapi/PythonAPI/pycocotools/concatenated/"+part+"border_table.json",
              "w")as outfile:
        json.dump(final_border_table, outfile, indent=4)

if __name__ == '__main__':
    part = "train"
    master(part)

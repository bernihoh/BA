import os
import numpy
from skimage import io
import matplotlib.pyplot as plt
import json
import pprint

img_annotation_json = json.load(open("/home/bernihoh/Bachelor/SMS/MaskRCNN/samples/SMSNetworks/face_feature_detection/via_project_5May2020_12h4m.json"))
via_img_data = img_annotation_json["_via_img_metadata"]
# print(via_img_data)
"""
new_dict:
(pic_id: { "pic_file_name": "str", 
           "pic_shape": [height, width, regions_length], 
           "regions": [{"annotation_polygon": {"all_points_x": [], 
                                               "all_points_y": []}, 
                        "annotation_class": "str"}]
"""
new_dict = {}
for img_data_key in via_img_data:
    img_data = via_img_data[img_data_key]
    pic_file_name = img_data["filename"]
    pic = io.imread("/home/bernihoh/Bachelor/SMS/MaskRCNN/samples/SMSNetworks/face_feature_detection/pic_val/"+pic_file_name)
    regions = img_data["regions"]
    regions_len = len(regions)
    pic_shape = [pic.shape[0], pic.shape[1], regions_len]
    new_regions_array = []
    for region in regions:
        all_points_x, all_points_y = region["shape_attributes"]["all_points_x"], region["shape_attributes"]["all_points_y"]
        annotation_polygon = {"all_points_x": all_points_x, "all_points_y": all_points_y}
        annotation_class = str(list(region["region_attributes"]["face_feature"].keys()))[2: -2]
        new_regions_array_item = {"annotation_polygon": annotation_polygon, "annotation_class": annotation_class}
        new_regions_array.append(new_regions_array_item)
    new_dict_item_value = {"pic_file_name": pic_file_name,
                           "pic_shape": pic_shape,
                           "regions": new_regions_array}
    new_dict_item_key = int(pic_file_name[0: -4])
    new_dict.update({new_dict_item_key: new_dict_item_value})

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(new_dict)
print(new_dict)
outfile = open("/home/bernihoh/Bachelor/SMS/MaskRCNN/samples/SMSNetworks/face_feature_detection/pic_val_annotations.json", "w")
json.dump(new_dict, outfile)



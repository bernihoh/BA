from json import dumps, load
import pprint
from typing import List, Any, Union
from skimage import io
from skimage.draw import line
from skimage import io
import os
import sys
import math
from math import *
import numpy as np
import pickle as pkl
import matplotlib
import matplotlib.pyplot as plt
import codecs
# from MaskRCNN.samples.SMSNetworks.SMSNetworks import FFN
from modification.helper_files import roi_helper
from modification.helper_files import math_helper
from multiprocessing import *


def calculate_diff_img(diff_list):
    width_one_side = 20
    diff_img = np.zeros((len(diff_list), 2*width_one_side+1))  # +1, because the 1 is the middle
    bs_counter = 0
    bs = 0
    for a, i in zip(diff_list, range(len(diff_list))):
        rel_place = (atan(a)/(math.pi/2))  # squishification function between -1 and 1
        abs_place = width_one_side + width_one_side*rel_place  # rel_place = 1 --> 40, rel_place = -1 --> 0, rel_place = 0 --> 20, even if you round
        if abs(width_one_side - abs_place) > 5:
            bs_counter += abs(width_one_side - abs_place)
        diff_img[i, int(round(abs_place))] = 1
        #diff_img[i, width_one_side] += 1
        #diff_img[i, width_one_side-5] += 1
        #diff_img[i, width_one_side+5] += 1
    if bs_counter >= 600:
        bs = 1
    return diff_img, bs_counter, bs


def face_score_input(face_score_net):
    """ face_score_net: {"points": {...}, "areas": {...}}
        ref_face_net: {"points": {...}, "areas": {...}}
    """
    """
    # This version includes the reference_nets
    face_score_input_dict = {
        "point_diff_values_x": {},
        "point_diff_values_y": {},
        "line_length_diff_values": {},
        "angle_diff_values": {},
        "area_diff_values": {},
        "flip_area_diff_values": {},
        "scaling_horizontal": 0,
        "scaling_vertical": 0,
        "degree_rot": 0
    }
    degree_rot = math_helper.calculate_rotation(face_score_net["points"]["mouth_point_array"])
    ref_face_net = load(open("/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/face_net_jsons_reference/face_net_" + str(np.round(degree_rot, 0)) + ".json"))
    face_score_input_dict["degree_rot"] = degree_rot
    face_score_input_dict["scaling_vertical"] = math_helper.calculate_line(ref_face_net["points"]["outer_eye_point_array_l"][0], ref_face_net["points"]["outer_eye_point_array_r"][-1])[1] / math_helper.calculate_line(face_score_net["points"]["outer_eye_point_array_l"][0], face_score_net["points"]["outer_eye_point_array_r"][-1])[1]
    face_score_input_dict["scaling_horizontal"] = math_helper.calculate_line(ref_face_net["points"]["eye_bridge_point"], ref_face_net["points"]["chin_point_array"][1])[1] / math_helper.calculate_line(face_score_net["points"]["eye_bridge_point"], face_score_net["points"]["chin_point_array"][1])[1]
    for area, i in zip(ref_face_net["areas"], range(0, len(ref_face_net["areas"]))):
        ref_face_net["areas"].update({area: np.array(ref_face_net["areas"][area])})
        roi_area = [roi_helper.find_outermost_line(ref_face_net["areas"][area], "up"), roi_helper.find_outermost_line(ref_face_net["areas"][area], "left"), roi_helper.find_outermost_line(ref_face_net["areas"][area], "down"), roi_helper.find_outermost_line(ref_face_net["areas"][area], "right")]
        cut_out_ref_area = roi_helper.cut_out_roi(ref_face_net["areas"][area], roi_area)
        face_score_input_dict["area_diff_values"].update({i: roi_helper.area_sum(roi_helper.find_area_diff_bool(cut_out_ref_area, face_score_net["areas"][area]))})
        face_score_input_dict["flip_area_diff_values"].update({i: roi_helper.area_sum(roi_helper.find_area_diff_bool(np.flip(cut_out_ref_area, 1), face_score_net["areas"][area]))})
    points = []
    points_ref = []
    for point_array in face_score_net["points"]:
        par = face_score_net["points"][point_array]
        par_ref = ref_face_net["points"][point_array]
        try:
            for point, ref_point in zip(par, par_ref):
                points.append(point)
                points_ref.append(ref_point)
        except:
            points.append(par)
            points_ref.append(par_ref)
    ck, cj, ci = 0, 0, 0
    for i in range(0, len(points)):
        for j in range(i+1, len(points)):
            for k in range(j+1, len(points)):
                angle = math_helper.calculate_angle(points[k], points[i], points[j])
                angle_ref = math_helper.calculate_angle(points_ref[k], points_ref[i], points_ref[j])
                face_score_input_dict["angle_diff_values"].update({ck: angle_ref - angle})
                ck += 1
            line_length = math_helper.calculate_line(points[i], points[j])[1]
            line_length_ref = math_helper.calculate_line(points_ref[i], points_ref[j])[1]
            face_score_input_dict["line_length_diff_values"].update({cj: line_length_ref - line_length})
            cj += 1
        point_vector = math_helper.calculate_line(points[i], points_ref[i])[0]
        face_score_input_dict["point_diff_values_x"].update({ci: point_vector[1]})
        face_score_input_dict["point_diff_values_y"].update({ci: point_vector[0]})
        ci += 1
    return face_score_input_dict
    """
    """
    # This version does not include the reference_nets
    face_score_input_dict = {
        "line_length_values": [],
        "angle_values": [],
        "areas": [],
        "degree_rot": 0
    }
    deg_rot = math_helper.calculate_rotation(face_score_net["points"]["mouth_point_array"])
    face_score_input_dict["degree_rot"] = deg_rot
    face_score_input_dict["areas"].append(roi_helper.roi_split(face_score_net["areas"]["iris_area_l"], 3, 3))
    face_score_input_dict["areas"].append(roi_helper.roi_split(face_score_net["areas"]["iris_area_r"], 3, 3))
    face_score_input_dict["areas"].append(roi_helper.roi_split(face_score_net["areas"]["inner_eye_area_l"], 4, 10))
    face_score_input_dict["areas"].append(roi_helper.roi_split(face_score_net["areas"]["inner_eye_area_r"], 4, 10))
    face_score_input_dict["areas"].append(roi_helper.roi_split(face_score_net["areas"]["outer_eye_area_l"], 8, 15))
    face_score_input_dict["areas"].append(roi_helper.roi_split(face_score_net["areas"]["outer_eye_area_r"], 8, 15))
    face_score_input_dict["areas"].append(roi_helper.roi_split(face_score_net["areas"]["mouth_area"], 10, 20))
    pts = []
    for pt_arr in face_score_net["points"]:
        if type(face_score_net["points"][pt_arr][0]) is list:
            pts = pts + face_score_net["points"][pt_arr]
        else:
            pts.append(face_score_net["points"][pt_arr])
    for p_0 in range(len(pts)):
        for p_1 in range(p_0+1, len(pts)):
            for p_2 in range(p_1+1, len(pts)):
                face_score_input_dict["angle_values"].append(math_helper.calculate_angle_deg(pts[p_0], pts[p_1], pts[p_2]))
            face_score_input_dict["line_length_values"].append(math_helper.vector_length(math_helper.calculate_vector(pts[p_0], pts[p_1])))
   # print("angles", len(face_score_input_dict["angle_values"]), "lines", len(face_score_input_dict["line_length_values"]),
    #      "areas", len(face_score_input_dict["areas"]), "degree_rot", 1)
    return face_score_input_dict
    """
    degree_rot = math_helper.calculate_rotation(face_score_net["points"]["mouth_point_array"])
    deg_range = 5
    degree_rot = degree_rot if abs(degree_rot) <= deg_range else -deg_range if degree_rot + deg_range < degree_rot - deg_range else deg_range
    print("degree:", degree_rot)
    ref_face_net = pkl.load(open("/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/face_net_pkl_ref/face_net_pkl_ref_" + str(int(np.round(degree_rot, 0))) + "_degree.pkl", 'rb'))
    pts = []
    pts_ref = []
    for pt_arr in face_score_net["points"]:
        if not (pt_arr == "iris_center_point_l" or pt_arr == "iris_center_point_r" or pt_arr == "eye_brow_point_array_l" or pt_arr == "eye_brow_point_array_r"):
            #print(pt_arr)
            if type(face_score_net["points"][pt_arr][0]) is list:
                pts = pts + face_score_net["points"][pt_arr]
                pts_ref = pts_ref + ref_face_net["points"][pt_arr]
            else:
                pts.append(face_score_net["points"][pt_arr])
                pts_ref.append(ref_face_net["points"][pt_arr])
    diff_l = []
    # for iris
    #print("a")
    rel_sr = math_helper.calculate_line(face_score_net["points"]["iris_center_point_l"], face_score_net["points"]["inner_eye_point_array_l"][0])[1] / math_helper.calculate_line(ref_face_net["points"]["iris_center_point_l"], ref_face_net["points"]["inner_eye_point_array_l"][0])[1]
    rel_sl = math_helper.calculate_line(ref_face_net["points"]["iris_center_point_l"], ref_face_net["points"]["inner_eye_point_array_l"][0])[1] / math_helper.calculate_line(face_score_net["points"]["iris_center_point_l"], face_score_net["points"]["inner_eye_point_array_l"][0])[1]
    if rel_sl > rel_sr:
        diff_l.append(-rel_sl)
    else:
        diff_l.append(+rel_sr)
    rel_sr = math_helper.calculate_line(face_score_net["points"]["iris_center_point_r"], face_score_net["points"]["inner_eye_point_array_r"][0])[1] / math_helper.calculate_line(ref_face_net["points"]["iris_center_point_r"], ref_face_net["points"]["inner_eye_point_array_r"][0])[1]
    rel_sl = math_helper.calculate_line(ref_face_net["points"]["iris_center_point_r"], ref_face_net["points"]["inner_eye_point_array_r"][0])[1] / math_helper.calculate_line(face_score_net["points"]["iris_center_point_r"], face_score_net["points"]["inner_eye_point_array_r"][0])[1]
    if rel_sl > rel_sr:
        diff_l.append(-rel_sl)
    else:
        diff_l.append(+rel_sr)

    # for eye_brows
    eb_arr = []
    for i in range(len(face_score_net["points"]["eye_brow_point_array_l"])):
        length = math_helper.calculate_line(face_score_net["points"]["eye_brow_point_array_l"][i], face_score_net["points"]["eye_bridge_point"])[1]
        r_length = math_helper.calculate_line(ref_face_net["points"]["eye_brow_point_array_l"][i], ref_face_net["points"]["eye_bridge_point"])[1]
        eb_arr.append([length, r_length])
    for i in range(len(face_score_net["points"]["eye_brow_point_array_r"])):
        length = math_helper.calculate_line(face_score_net["points"]["eye_brow_point_array_r"][i], face_score_net["points"]["eye_bridge_point"])[1]
        r_length = math_helper.calculate_line(ref_face_net["points"]["eye_brow_point_array_r"][i],  ref_face_net["points"]["eye_bridge_point"])[1]
        eb_arr.append([length, r_length])
    for length, r_length in eb_arr:
        try:
            rel_sr = length / r_length
        except ZeroDivisionError:
            rel_sr = 0
        try:
            rel_sl = r_length / length
        except ZeroDivisionError:
            rel_sl = 0
        if rel_sl > rel_sr:
            diff_l.append(-rel_sl)
        else:
            diff_l.append(rel_sr)
    # print("b")
    for i, ri in zip(range(len(pts)), range(len(pts_ref))):
        for j, rj in zip(range(i+1, len(pts)), range(ri+1, len(pts_ref))):
            length = math_helper.calculate_line(pts[i], pts[j])[1]
            r_length = math_helper.calculate_line(pts_ref[ri], pts_ref[rj])[1]
            try:
                rel_sr = length/r_length
            except ZeroDivisionError:
                rel_sr = 1000
            try:
                rel_sl = r_length/length
            except ZeroDivisionError:
                rel_sl = 1000
            if rel_sl > rel_sr:
                diff_l.append(-rel_sl)
            else:
                diff_l.append(rel_sr)
    #print("d")
    avg_dl = sum(diff_l)/len(diff_l)
    print("-----------------------avg", avg_dl)
    avg_l = [a-avg_dl for a in diff_l]

    fsi = calculate_diff_img(avg_l)

    return fsi


def create_face_net_picture_out_of_dict(face_net_dictionary, picture_shape):
    face_net_picture = np.zeros(picture_shape)
    # iris_l
    print("a")
    iris_area_l = face_net_dictionary['iris_area_l']
    anchor_iris_l = [int(a-b) for a, b in zip(face_net_dictionary['iris_center_point_l'], [iris_area_l.shape[0]/2, iris_area_l.shape[1]/2])]
    face_net_picture = draw_image_on_canvas(face_net_picture, iris_area_l, anchor_iris_l)
    # iris_r
    iris_area_r = face_net_dictionary['iris_area_r']
    anchor_iris_r = [int(a-b) for a, b in zip(face_net_dictionary['iris_center_point_r'], [iris_area_r.shape[0]/2, iris_area_r.shape[1]/2])]
    face_net_picture = draw_image_on_canvas(face_net_picture, iris_area_r, anchor_iris_r)
    # inner_eye_l
    inner_eye_area_l = face_net_dictionary['inner_eye_area_l']
    left_centric_point = roi_helper.find_side_centric_point(inner_eye_area_l, "left")
    anchor_inner_eye_l = [int(a-b) for a, b in zip(face_net_dictionary['inner_eye_point_array_l'][0], left_centric_point)]
    face_net_picture = draw_image_on_canvas(face_net_picture, inner_eye_area_l, anchor_inner_eye_l)
    line_img, line_img_anchor = create_line_image(face_net_dictionary['inner_eye_point_array_l'][0], face_net_dictionary['iris_center_point_l'])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(face_net_dictionary['inner_eye_point_array_l'][-1], face_net_dictionary['iris_center_point_l'])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    # inner_eye_r
    inner_eye_area_r = face_net_dictionary['inner_eye_area_r']
    left_centric_point = roi_helper.find_side_centric_point(inner_eye_area_r, "left")
    anchor_inner_eye_r = [int(a-b) for a, b in zip(face_net_dictionary['inner_eye_point_array_r'][0], left_centric_point)]
    face_net_picture = draw_image_on_canvas(face_net_picture, inner_eye_area_r, anchor_inner_eye_r)
    line_img, line_img_anchor = create_line_image(face_net_dictionary['inner_eye_point_array_r'][0], face_net_dictionary['iris_center_point_r'])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(face_net_dictionary['inner_eye_point_array_r'][-1], face_net_dictionary['iris_center_point_r'])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    # outer_eye_l
    outer_eye_area_l = face_net_dictionary['outer_eye_area_l']
    left_centric_point = roi_helper.find_side_centric_point(outer_eye_area_l, "left")
    anchor_outer_eye_l = [int(a-b) for a, b in zip(face_net_dictionary['outer_eye_point_array_l'][0], left_centric_point)]
    face_net_picture = draw_image_on_canvas(face_net_picture, outer_eye_area_l, anchor_outer_eye_l)
    line_img, line_img_anchor = create_line_image(face_net_dictionary['outer_eye_point_array_l'][0], face_net_dictionary['inner_eye_point_array_l'][0])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(face_net_dictionary['outer_eye_point_array_l'][-1], face_net_dictionary['inner_eye_point_array_l'][-1])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    # outer_eye_r
    outer_eye_area_r = face_net_dictionary['outer_eye_area_r']
    left_centric_point = roi_helper.find_side_centric_point(outer_eye_area_r, "left")
    anchor_outer_eye_r = [int(a-b) for a, b in zip(face_net_dictionary['outer_eye_point_array_r'][0], left_centric_point)]
    face_net_picture = draw_image_on_canvas(face_net_picture, outer_eye_area_r, anchor_outer_eye_r)
    line_img, line_img_anchor = create_line_image(face_net_dictionary['outer_eye_point_array_r'][0], face_net_dictionary['inner_eye_point_array_r'][0])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(face_net_dictionary['outer_eye_point_array_r'][-1], face_net_dictionary['inner_eye_point_array_r'][-1])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    # mouth
    mouth_area = face_net_dictionary['mouth_area']
    #plt.imshow(mouth_area)
    #plt.show()
    left_centric_point = roi_helper.find_side_centric_point(mouth_area, "left")
    anchor_mouth = [int(a-b) for a, b in zip(face_net_dictionary['mouth_point_array'][0], left_centric_point)]
    #print(anchor_mouth)
    face_net_picture = draw_image_on_canvas(face_net_picture, mouth_area, anchor_mouth)
    # eye_brow_l
    eye_brow_point_array_l = face_net_dictionary['eye_brow_point_array_l']
    for i in range(len(eye_brow_point_array_l)-1):
        line_img, line_img_anchor = create_line_image(eye_brow_point_array_l[i], eye_brow_point_array_l[i+1])
        face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    # eye_brow_r
    eye_brow_point_array_r = face_net_dictionary['eye_brow_point_array_r']
    for i in range(len(eye_brow_point_array_r)-1):
        line_img, line_img_anchor = create_line_image(eye_brow_point_array_r[i], eye_brow_point_array_r[i+1])
        face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    # eye_bridge_point
    eye_bridge_point = face_net_dictionary['eye_bridge_point']
    line_img, line_img_anchor = create_line_image(eye_brow_point_array_l[-1], eye_bridge_point)
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(eye_brow_point_array_r[0], eye_bridge_point)
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(face_net_dictionary['outer_eye_point_array_l'][-1], eye_bridge_point)
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(face_net_dictionary['outer_eye_point_array_r'][0], eye_bridge_point)
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    # cheek_l
    cheek_point_array_l = face_net_dictionary['cheek_point_array_l']
    line_img, line_img_anchor = create_line_image(cheek_point_array_l[0], cheek_point_array_l[1])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(cheek_point_array_l[0], face_net_dictionary['outer_eye_point_array_l'][0])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(cheek_point_array_l[1], face_net_dictionary['outer_eye_point_array_l'][0])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(cheek_point_array_l[1], face_net_dictionary['outer_eye_point_array_l'][-1])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    # cheek_r
    cheek_point_array_r = face_net_dictionary['cheek_point_array_r']
    line_img, line_img_anchor = create_line_image(cheek_point_array_r[0], cheek_point_array_r[1])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(cheek_point_array_r[1], face_net_dictionary['outer_eye_point_array_r'][-1])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(cheek_point_array_r[0], face_net_dictionary['outer_eye_point_array_r'][0])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(cheek_point_array_r[0], face_net_dictionary['outer_eye_point_array_r'][-1])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    # nose
    nose_tip_point_array = face_net_dictionary['nose_tip_point_array']
    line_img, line_img_anchor = create_line_image(nose_tip_point_array[0], nose_tip_point_array[-1])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(nose_tip_point_array[0], nose_tip_point_array[1])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(nose_tip_point_array[1], nose_tip_point_array[-1])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(nose_tip_point_array[0], eye_bridge_point)
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(nose_tip_point_array[1], eye_bridge_point)
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(nose_tip_point_array[-1], eye_bridge_point)
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(nose_tip_point_array[0], face_net_dictionary['outer_eye_point_array_l'][-1])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(nose_tip_point_array[0], cheek_point_array_l[-1])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(nose_tip_point_array[-1], face_net_dictionary['outer_eye_point_array_r'][0])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(nose_tip_point_array[-1], cheek_point_array_r[0])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    # mouth
    mouth_point_array = face_net_dictionary['mouth_point_array']
    line_img, line_img_anchor = create_line_image(mouth_point_array[0], mouth_point_array[-1])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(mouth_point_array[0], mouth_point_array[1])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(mouth_point_array[1], mouth_point_array[-1])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(mouth_point_array[0], face_net_dictionary['outer_eye_point_array_l'][0])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(mouth_point_array[0], cheek_point_array_l[-1])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(mouth_point_array[0], nose_tip_point_array[0])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(mouth_point_array[1], nose_tip_point_array[1])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(mouth_point_array[-1], face_net_dictionary['outer_eye_point_array_r'][-1])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(mouth_point_array[-1], cheek_point_array_r[0])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(mouth_point_array[-1], nose_tip_point_array[-1])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    # chin
    chin_point_array = face_net_dictionary['chin_point_array']
    line_img, line_img_anchor = create_line_image(chin_point_array[0], chin_point_array[-1])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(chin_point_array[0], chin_point_array[1])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(chin_point_array[1], chin_point_array[-1])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(chin_point_array[0], nose_tip_point_array[0])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(chin_point_array[0], mouth_point_array[0])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(chin_point_array[-1], nose_tip_point_array[-1])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(chin_point_array[-1], mouth_point_array[-1])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    line_img, line_img_anchor = create_line_image(chin_point_array[1], mouth_point_array[1])
    face_net_picture = draw_image_on_canvas(face_net_picture, line_img, line_img_anchor)
    print("b")
    #plt.imshow(face_net_picture)
    #plt.show()
    return face_net_picture


def create_line_image(point_start, point_end):
    point_start = [int(np.round(point_start[0])), int(np.round(point_start[1]))]
    point_end = [int(np.round(point_end[0])), int(np.round(point_end[1]))]
    line_img = np.zeros((abs(point_start[0] - point_end[0]) + 1, abs(point_start[1] - point_end[1]) + 1), dtype=np.float64)
    line_img_anchor = [min(point_start[0], point_end[0]), min(point_start[1], point_end[1])]
    rr, cc = line(point_start[0] - line_img_anchor[0], point_start[1] - line_img_anchor[1], point_end[0] - line_img_anchor[0], point_end[1] - line_img_anchor[1])
    line_img[rr, cc] = 1
    return [line_img, line_img_anchor]


def draw_image_on_canvas(canvas, image, anchor_point):
    """'image' will be drawn into the 'canvas'-image. The 'anchor_point' is the left-upper point where 'image' starts in the 'canvas'-image"""
    for i in range(image.shape[0]):
        canvas[i+anchor_point[0], anchor_point[1]: anchor_point[1]+image.shape[1]] += image[i]
    return canvas


def draw_image_from_dict_file(picture_path):
    pkl_file = open(picture_path, 'rb')
    face_net_score_dict = pkl.load(pkl_file)
    pkl_file.close()
    face_net_dict = face_net_score_dict["points"]
    face_net_dict.update(face_net_score_dict["areas"])
    picture_shape = (256, 256)  # for reference-masks (480, 510), for stylegan2-masks (256, 256)
    face_net_picture = create_face_net_picture_out_of_dict(face_net_dict, picture_shape)
    return face_net_picture


def parallel_face_score_input(pickle_files_list, p_num, path):
    for file_name, i in zip(pickle_files_list, range(len(pickle_files_list))):
        pkl_file = open(path + file_name, 'rb')
        face_net_score_dict = pkl.load(pkl_file)

        pkl_file.close()
        face_score_input_dict = face_score_input(face_net_score_dict)
        if p_num == 0:
            plt.imshow(face_score_input_dict)
            plt.title(file_name)
            plt.show()
        print(p_num)

        save_path = "/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/face_score_input_train/"+file_name
        file = open(save_path, 'wb')
        pkl.dump(face_score_input_dict, file)
        file.close()

    return


def parallel_draw_image_from_dict_file(pickle_files_list, p_num, path):
    for file_name, i in zip(pickle_files_list, range(len(pickle_files_list))):
        net_pic = draw_image_from_dict_file(path + file_name)
        save_path = "/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/face_net_pics_val/"+file_name[: -4]+".png"
        io.imsave(save_path, net_pic)
        print(p_num)


if __name__ == "__main__":
    """
    pickle_files_path = "/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/face_net_pkl_ref/"
    pickle_files = os.listdir(pickle_files_path)

    for pickle_file, i in zip(pickle_files, range(len(pickle_files))):
        print("Progress:", i+1, "/", len(pickle_files))
        pic_name = pickle_file[:-4]
        try:
            face_net_picture = draw_image_from_dict_file(pickle_files_path + pickle_file)
            io.imsave(
                "/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/ref_face_net_pic/" + pic_name + ".png",
                face_net_picture)

        except:
            pass
    """
    """
    pickle_files_path = "/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/face_net_pkl_ref/"
    pickle_file = "face_net_pkl_ref_0_degree.pkl"
    pic_name = pickle_file[:-4]
    face_net_picture = draw_image_from_dict_file(pickle_files_path + pickle_file)
    io.imsave("/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/ref_face_net_pic/" + pic_name + ".png", face_net_picture)
    """
    """
    pickle_file = "/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/face_net_jsons/0.png-mirror.pkl"
    pic_name = "0.png-mirror"
    face_net_picture = draw_image_from_dict_file(pickle_file)
    io.imsave("/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/face_net_score_pics/"+pic_name+".png", face_net_picture)
    """
    pickle_file_path = "/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/face_net_pkl_train/"
    pickle_files = os.listdir(pickle_file_path)
    truth_array = np.zeros((2, len(pickle_files)))
    for pickle_file_name, i in zip(pickle_files, range(len(pickle_files))):
        print("Progress:", i + 1, "/", len(pickle_files), pickle_file_name)
        pkl_file = open(pickle_file_path+pickle_file_name, 'rb')
        face_net_score_dict = pkl.load(pkl_file)
        pkl_file.close()
        face_score_input_dict, bs_amount, bs_value = face_score_input(face_net_score_dict)
        plt.imshow(face_score_input_dict)
        plt.title(pickle_file_name[0: 2]+"_bsamount_"+str(bs_amount)+"_bsvalue_"+str(bs_value))
        plt.show()
        try:
            truth_array[0, i] = int(pickle_file_name[0: 3])
            score_name = pickle_file_name[0: 3]
        except:
            try:
                truth_array[0, i] = int(pickle_file_name[0: 2])
                score_name = pickle_file_name[0: 2]
            except:
                truth_array[0, i] = int(pickle_file_name[0: 1])
                score_name = pickle_file_name[0: 1]

        truth_array[1, i] = bs_value
        pic_save_file_path = "/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/face_score_input_train/"
        pic_save_file = open(pic_save_file_path+score_name+".pkl", 'wb')
        pkl.dump(face_score_input_dict, pic_save_file)
        pic_save_file.close()
    print(truth_array)
    ta_save_path = "/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/"
    file = open(ta_save_path + "truth_train.pkl", 'wb')
    pkl.dump(truth_array, file)
    file.close()
    """
    truth_path = "/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/truth_train.pkl"
    truth_file = open(truth_path, 'rb')
    truth_array = pkl.load(truth_file)
    sum = 0
    for i in truth_array[1]:
        assert 0<=i<=1
        sum += i
    print(sum)
    """
    """
    pickle_file_path = "/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/face_net_pkl_train/"
    pickle_files = os.listdir(pickle_file_path)
    pool = Pool(cpu_count())
    chunk_length = int(len(pickle_files) / cpu_count())
    for i in range(cpu_count()):
        pickle_files_chunk = pickle_files[i * chunk_length: (i+1) * chunk_length] if not i == 31 else pickle_files[i * chunk_length: len(pickle_files)]
        pool.apply_async(parallel_face_score_input, args=(pickle_files_chunk, i, pickle_file_path))
    pool.close()
    """
    """
    pickle_file_path = "/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/face_net_pkl_val/"
    pickle_files = os.listdir(pickle_file_path)
    pool = Pool(cpu_count())
    chunk_length = int(len(pickle_files) / cpu_count())
    for i in range(cpu_count()):
        pickle_files_chunk = pickle_files[i * chunk_length: (i+1) * chunk_length] if not i == 31 else pickle_files[i * chunk_length: len(pickle_files)]
        pool.apply_async(parallel_draw_image_from_dict_file, args=(pickle_files_chunk, i, pickle_file_path))
    pool.close()
    """


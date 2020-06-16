import random
from typing import Any
import numpy as np
from math import pi, sin, cos
from skimage.draw import polygon
from skimage import io
import cv2
import math
import modification.icp.icp as icp
import MaskRCNN.samples.SMSNetworks.SMSNetworks as SMSNetworks
from stylegan2.run_generator import generate_imgs as generate_heads
import modification.netcompare.netcompare as netcompare
from modification.pic_fitting_tool.pic_fitting import pic_fitting, edge_detection
import modification.helper_files.roi_helper as roi_helper
import modification.helper_files.math_helper as math_helper
import matplotlib.pyplot as plt
import tensorflow as tf


def array_harmonizing(va_0,  va_1, l_s, r_s, arr):

    """calculates the harmonizing color flow of the array 'arr' between the pt-value 'va_0' left of 'arr' and pt-value 'va_1' right of 'arr'."""

    ls_pt = len(arr) * l_s
    rs_pt = len(arr) * r_s
    hrzsc_l = (pi/2) / (l_s * len(arr)) if l_s > 0 else 0
    hrzsc_r = (pi/2) / (len(arr) - r_s*len(arr)) if r_s < 1 else 0
    sin2_l = lambda x: sin(hrzsc_l*x)**2
    sin2_r = lambda x: sin(hrzsc_r*x)**2
    cos2_l = lambda x: cos(hrzsc_l*x)**2
    cos2_r = lambda x: cos(hrzsc_r*x)**2
    h_arr = np.zeros(arr.shape)
    if len(arr.shape) > 2:
        for i in range(arr.shape[0]):
            if i <= ls_pt:
                h_arr[i] = np.asarray([va_0i*cos2_l(i) + v_arri*sin2_l(i) for va_0i, v_arri in zip(va_0, arr[i])])
            elif ls_pt < i <= rs_pt:
                h_arr[i] = arr[i]
            else:
                h_arr[i] = np.asarray([va_1i*sin2_r(i-rs_pt) + v_arri*cos2_r(i-rs_pt) for va_1i, v_arri in zip(va_1, arr[i])])

    else:
        for i in range(arr.shape[0]):
            if i <= ls_pt:
                h_arr[i] = va_0*cos2_l(i) + arr[i]*sin2_l(i)
            elif ls_pt < i <= rs_pt:
                h_arr[i] = arr[i]
            else:
                h_arr[i] = va_1*sin2_r(i-rs_pt) + arr[i]*cos2_r(i-rs_pt)

    #h_arr = arr
    return h_arr


class head_creation:
    def __init__(self, label_map, spade_image):
        """the SPADE image, that gets further improvements"""
        self.spade_img = cv2.resize(spade_image, (256, 256))
        """the segmentation map drawn from the user"""
        self.label_map = cv2.resize(label_map, (256, 256))

    def head_creator(self):
        head_detection = SMSNetworks.HDN(self.label_map)
        hdn_result = head_detection.detect()  # {"masks": ,"rois": }
        head_generator = generate_heads()
        face_net_comp = netcompare.face_net_compare(service="face_net")
        return_arr = []
        for i in range(len(hdn_result["rois"])):
            """
            hdn_head = roi_helper.cut_out_roi(hdn_result["masks"][i],
                                              (roi_helper.find_outermost_line(hdn_result["masks"][i], "left"),
                                              roi_helper.find_outermost_line(hdn_result["masks"][i], "up"),
                                              roi_helper.find_outermost_line(hdn_result["masks"][i], "down"),
                                              roi_helper.find_outermost_line(hdn_result["masks"][i], "right")))
            """
            hdn_head = roi_helper.cut_out_roi(hdn_result["masks"][:, :, i], hdn_result["rois"][i])
            hdn_roi = hdn_result["rois"][i]

            while True:
                try:
                    seed, trunc_psi = random.randint(0, 2000), random.randint(0, 100) / 100
                    #gen_head = head_generator.generate_images(seed, trunc_psi)[0]
                    #gen_head = io.imread("/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/face_pics_val/19.png")
                    gen_head = io.imread("/home/bernihoh/Bachelor/SMS/modification/netcompare/services/face_net/face_pics_val/37.png")
                    ffd = SMSNetworks.FFN(gen_head)
                    ffn_res = ffd.detect()
                    bs = face_net_comp.compare_net(ffn_res["face_score_input"])
                    if not bs:
                        break
                except:
                    pass

            face_net = ffn_res["face_net"]
            head_mask = ffn_res["head_mask"]
            face_mask = ffn_res["face_mask"]
            # face_less_mask
            rows, cols = [], []
            oe_l, oe_r = face_net["outer_eye_point_array_l"][0], face_net["outer_eye_point_array_r"][-1],
            rows.append(oe_l[0]), cols.append(oe_l[1])
            for c in range(oe_l[1], oe_r[1]):
                r = roi_helper.find_most_outer_points_of_array(face_mask[:, c])[0]
                rows.append(r), cols.append(c)
            rows.append(oe_r[0]), cols.append(oe_r[1])
            rows.append(face_net["mouth_point_array"][-1][0]), cols.append(face_net["mouth_point_array"][-1][1])
            rows.append(face_net["chin_point_array"][-1][0]), cols.append(face_net["chin_point_array"][-1][1])
            rows.append(face_net["chin_point_array"][0][0]), cols.append(face_net["chin_point_array"][0][1])
            rows.append(face_net["mouth_point_array"][0][0]), cols.append(face_net["mouth_point_array"][0][1])
            face_less_mask = np.zeros((gen_head.shape[0], gen_head.shape[1]), dtype=np.uint8)
            rr, cc = polygon(np.asarray(rows), np.asarray(cols))
            face_less_mask[rr, cc] = 1
            co_face = np.zeros(gen_head.shape)
            """
            for j in range(len(gen_head.shape[2])):
                co_face[:, :, j] = gen_head[:, :, j] * face_less_mask
            """
            co_head = np.zeros(gen_head.shape)
            for j in range(gen_head.shape[2]):
                co_head[:, :, j] = gen_head[:, :, j] * head_mask
            """
            co_face = resize(roi_helper.cut_out_roi(co_face, face_net["head_roi"]),
                             hdn_head.shape, cval=0, anti_aliasing=True)
            """
            face_less_mask = cv2.resize(roi_helper.cut_out_roi(face_less_mask, ffn_res["head_roi"]), (hdn_head.shape[1], hdn_head.shape[0]))
            co_head = cv2.resize(roi_helper.cut_out_roi(co_head, ffn_res["head_roi"]), (hdn_head.shape[1], hdn_head.shape[0]), interpolation=cv2.INTER_AREA)
            head_mask = cv2.resize(roi_helper.cut_out_roi(head_mask, ffn_res["head_roi"]), (hdn_head.shape[1], hdn_head.shape[0]))
            plt.imshow(face_less_mask)
            plt.title("face_less_mask")
            plt.show()
            plt.imshow(co_head)
            plt.title("co_head")
            plt.show()
            plt.imshow(head_mask)
            plt.title("head_mask")
            plt.show()

            spade_head_place = self.spade_img
            spade_head_place = roi_helper.cut_out_roi(spade_head_place, hdn_roi)
            for j in range(spade_head_place.shape[2]):
                spade_head_place[:, :, j] = spade_head_place[:, :, j] * hdn_head
            plt.imshow(spade_head_place)
            plt.title("spade_head_place")
            plt.show()
            # do icp for hdn_result["masks"][i] and head_mask <-- thats correct
            # create Point-Cloud for hdn_result["masks"][i]
            hdn_pt_cl = []
            for y in range(hdn_head.shape[0]):
                for x in range(hdn_head.shape[1]):
                    if np.sum(hdn_head[y, x]) > 0:
                        hdn_pt_cl.append([y, x])
            hm_pt_cl = []
            for y in range(head_mask.shape[0]):
                for x in range(head_mask.shape[1]):
                    if np.sum(head_mask[y, x]) > 0:
                        hm_pt_cl.append([y, x])
            trans_dir = np.asarray([1, 1, 1]) if len(hm_pt_cl) > len(hdn_pt_cl) else np.asarray([1, 1, -1])
            scene_pt_cl = hdn_pt_cl if len(hm_pt_cl) > len(hdn_pt_cl) else hm_pt_cl
            model_pt_cl = hm_pt_cl if len(hm_pt_cl) > len(hdn_pt_cl) else hdn_pt_cl

            T, distances, iterations = icp.icp(np.asarray(model_pt_cl), np.asarray(scene_pt_cl), tolerance=0.000001)
            print(T, distances, iterations, "Rotation")
            proj_hm = np.zeros(hdn_head.shape)
            proj_cohd = np.zeros((hdn_head.shape[0], hdn_head.shape[1], 3), dtype=np.uint8)
            proj_flm = np.zeros(hdn_head.shape)
            for y, x in model_pt_cl:
                v = np.asarray([y, x, 1])
                T_dir = np.multiply(T, trans_dir)
                v_p = T_dir.dot(v)[0: -1]
                p_y, p_x = int(np.round(v_p[0])), int(np.round(v_p[1]))
                try:
                    proj_hm[p_y, p_x] = 1 if np.sum(head_mask[y, x]) > 0 and p_y >= 0 and p_x >= 0 else 0
                    proj_cohd[p_y, p_x, :] = co_head[y, x, :] if p_y >= 0 and p_x >= 0 else np.asarray([0, 0, 0])
                    proj_flm[p_y, p_x] = 1 if np.sum(face_less_mask[y, x]) > 0 else 0
                except:
                    pass
            for y in range(proj_cohd.shape[0]):
                for x in range(1, proj_cohd.shape[1]-1):
                    if np.sum(proj_cohd[y, x, :]) == 0:
                        if np.sum(proj_cohd[y, x-1, :]) > 0 and np.sum(proj_cohd[y, x+1, :]) > 0:
                                proj_cohd[y, x, :] = proj_cohd[y, x+1, :]
                                #print(co_head[y, x])

            plt.imshow(proj_hm)
            plt.title("proj_hm")
            plt.show()
            plt.imshow(proj_cohd)
            plt.title("proj_cohd")
            plt.show()
            plt.imshow(proj_flm)
            plt.title("proj_flm")
            plt.show()
            for y in range(proj_cohd.shape[0]):
                a_1, b_0 = roi_helper.find_most_outer_points_of_array(proj_flm[y])
                if not a_1 == b_0 == 0:
                    a_0, b_1 = roi_helper.find_most_outer_points_of_array(proj_hm[y])
                    a_0 = max(0, a_0-1)
                    b_1 = min(len(proj_hm[y])-1, b_1+1)
                    a_arr = proj_cohd[y, a_0: a_1]
                    b_arr = proj_cohd[y, b_0: b_1]
                    va_0, va_1, vb_0, vb_1 = self.spade_img[y+hdn_roi[0], a_0+hdn_roi[1]], proj_cohd[y, a_1], proj_cohd[y, b_0], self.spade_img[y+hdn_roi[0], b_1+hdn_roi[1]]
                    proj_cohd[y, a_0: a_1] = array_harmonizing(va_0, va_1, 0.4, 1.0, a_arr)
                    proj_cohd[y, b_0: b_1] = array_harmonizing(vb_0, vb_1, 0.0,  0.6, b_arr)
                else:
                    a_0, a_1 = roi_helper.find_most_outer_points_of_array(proj_hm[y])
                    a_0 = max(0, a_0-1)
                    a_1 = min(len(proj_hm[y])-1, a_1+1)
                    a_arr = proj_cohd[y, a_0: a_1]
                    va_0, va_1 = self.spade_img[y+hdn_roi[0], a_0+hdn_roi[1]], self.spade_img[y+hdn_roi[0], a_1+hdn_roi[1]]
                    proj_cohd[y, a_0: a_1] = array_harmonizing(va_0, va_1, 0.15, 0.85, a_arr)

            for x in range(proj_cohd.shape[1]):
                a_1, b_0 = roi_helper.find_most_outer_points_of_array(proj_flm[:, x])
                if not a_1 == b_0 == 0:
                    a_0, b_1 = roi_helper.find_most_outer_points_of_array(proj_hm[:, x])
                    a_0 = max(0, a_0-1)
                    b_1 = min(len(proj_hm[:, x])-1, b_1+1)
                    a_arr = proj_cohd[a_0: a_1, x]
                    b_arr = proj_cohd[b_0: b_1, x]
                    va_0, va_1, vb_0, vb_1 = self.spade_img[a_0+hdn_roi[0], x+hdn_roi[1]], proj_cohd[a_1, x], proj_cohd[b_0, x], self.spade_img[b_1+hdn_roi[0], x+hdn_roi[1]]
                    proj_cohd[a_0: a_1, x] = array_harmonizing(va_0, va_1, 0.15, 1.0, a_arr)
                    proj_cohd[b_0: b_1, x] = array_harmonizing(vb_0, vb_1, 0.0, 0.85, b_arr)
                else:
                    a_0, a_1 = roi_helper.find_most_outer_points_of_array(proj_hm[:, x])
                    a_0 = max(0, a_0 - 1)
                    a_1 = min(len(proj_hm[:, x]) - 1, a_1 + 1)
                    a_arr = proj_cohd[a_0: a_1, x]
                    va_0, va_1 = self.spade_img[a_0 + hdn_roi[0], x + hdn_roi[1]], self.spade_img[a_1 + hdn_roi[0], x + hdn_roi[1]]
                    proj_cohd[a_0: a_1, x] = array_harmonizing(va_0, va_1, 0.07, 0.93, a_arr)


            plt.imshow(proj_cohd)
            plt.title("harmonized proj_cohd")
            plt.show()
            for y in range(spade_head_place.shape[0]):
                for x in range(spade_head_place.shape[1]):
                    if np.sum(proj_cohd[y, x]) > 0:
                        spade_head_place[y, x] = proj_cohd[y, x]
            for y in range(hdn_roi[0], hdn_roi[2]):
                for x in range(hdn_roi[1], hdn_roi[3]):
                    if np.sum(spade_head_place[y - hdn_roi[0], x - hdn_roi[1]]) > 0:
                        self.spade_img[y, x] = spade_head_place[y - hdn_roi[0], x - hdn_roi[1]]
            #self.spade_img[hdn_roi[0]: hdn_roi[2], hdn_roi[1]: hdn_roi[3]] = spade_head_place
            #return_arr.append(spade_head_place)
        return self.spade_img


if __name__ == "__main__":
    name = "2020-06-06 20:04:03.png"
    spade_img = io.imread("/home/bernihoh/Bachelor/SPADE/results/coco_pretrained/test_latest/images/synthesized_image/"+name)
    label_map = io.imread("/home/bernihoh/Bachelor/SPADE/results/coco_pretrained/test_latest/images/input_label/"+name)
    head_creator = head_creation(label_map, spade_img)
    spade_img = head_creator.head_creator()
    plt.imshow(spade_img)
    plt.show()


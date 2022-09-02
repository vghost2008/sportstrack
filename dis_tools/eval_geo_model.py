import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import wml_utils as wmlu
import os.path as osp
import object_detection2.bboxes as odb
from keypoints.get_keypoints import KPDetection
from sportstracker.matching import *
import sys

sys.path.append("..")

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

class IOUModel:
    def __init__(self) -> None:
        pass

    def compute_distance_matrix(self,feature0,feature1):
        img0,bboxes0 = feature0
        img1,bboxes1 = feature1

        ious = odb.iou_matrix(bboxes0,bboxes1)
        return 1.0-ious

class GIOUModel:
    def __init__(self) -> None:
        pass

    def compute_distance_matrix(self,feature0,feature1):
        img0,bboxes0 = feature0
        img1,bboxes1 = feature1

        gious = odb.giou_matrix(bboxes0,bboxes1)
        return 1-(gious+1)/2

class LTrack:
    def __init__(self,bbox,kps):
        self.ltrb = bbox
        self.cur_kps = kps
    
    @staticmethod
    def make_tracks(bboxes,kps):
        res = []
        for i in range(len(bboxes)):
            res.append(LTrack(bboxes[i],kps[i]))
        return res

class KeypointModel:
    def __init__(self) -> None:
        self.model = KPDetection()

    def compute_distance_matrix(self,feature0,feature1):
        img0,bboxes0 = feature0
        img1,bboxes1 = feature1
        kps0 = self.model.get_kps_by_bboxes(img0,bboxes0,scale_bboxes=True)
        kps1 = self.model.get_kps_by_bboxes(img1,bboxes1,scale_bboxes=True)
        tracks0 = LTrack.make_tracks(bboxes0,kps0)
        tracks1 = LTrack.make_tracks(bboxes1,kps1)
        return kps_distancev2(tracks0, tracks1)


class EvalGeoDis:
    def __init__(self, model, eval_dir, save_fig_dir):
        self.model = model
        self.eval_dir = eval_dir
        self.save_fig_dir = save_fig_dir
        self.pos = []
        self.neg = []
        self.type = ""

    def get_dir(self):
        dir_list = os.listdir(self.eval_dir)
        return dir_list

    @staticmethod
    def txt_to_dict(txt_path):
        with open(txt_path, encoding="utf-8") as f:
            content = f.readlines()
        info_dict = {}
        for line in content:
            f_id, p_id, xmin, ymin, w, h, _, _, _ = [int(x) for x in line.strip().split(",")]
            if f_id not in info_dict:
                info_dict[f_id] = {}
                info_dict[f_id][p_id] = [xmin, ymin, w, h]
            else:
                info_dict[f_id][p_id] = [xmin, ymin, w, h]
        return info_dict

    def get_features(self, img_dir, gt_dict, frame_idx):
        image_1 = cv2.imread(os.path.join(img_dir, str(frame_idx).zfill(6) + ".jpg"))
        image_1 = image_1[...,::-1]
        bboxes = self.get_person_bbox_list(image_1, gt_dict, frame_idx)
        return (image_1,bboxes)

    def process(self):
        dir_list = self.get_dir()
        threshold = 0.95
        for _i, name in enumerate(dir_list):
            print(f"{name} start {_i}/{len(dir_list)}")
            img_dir = os.path.join(self.eval_dir, name, "img1")
            gt_path = os.path.join(self.eval_dir, name, "gt", "gt.txt")
            gt_dict = self.txt_to_dict(gt_path)
            all_keys = sorted(gt_dict.keys())
            frame_idx0 = all_keys[0]
            last_frame_id = frame_idx0
            last_features = self.get_features(img_dir, gt_dict, frame_idx0)
            for frame_id in all_keys[1:]:
                person_reid_list1 = list(gt_dict[last_frame_id].keys())
                person_reid_list2 = list(gt_dict[frame_id].keys())
                person_feature1 = last_features
                person_feature2 = self.get_features(img_dir, gt_dict, frame_id)

                distmat = self.model.compute_distance_matrix(person_feature1,person_feature2)
                for r_id1 in person_reid_list1:
                    idx1 = person_reid_list1.index(r_id1)
                    if r_id1 in person_reid_list2:
                        idx2 = person_reid_list2.index(r_id1)
                        pos = distmat[idx1][idx2]
                        self.pos.append(pos)
                        if self.type == "mean":
                            neg = (np.sum(distmat[idx1, :]) - pos) / (distmat.shape[1] - 1)
                            self.neg.append(neg)
                        else:
                            if distmat.shape[1] > 1:
                                for i, v in enumerate(distmat[idx1, :]):
                                    if i == idx2 or v>=threshold:
                                        continue
                                    self.neg.append(v)

                    else:
                        if self.type == "mean":
                            neg = (np.sum(distmat[idx1, :])) / (distmat.shape[1])
                            self.neg.append(neg)
                        else:
                            for i, v in enumerate(distmat[idx1, :]):
                                if v<threshold:
                                    self.neg.append(v)
                last_features = person_feature2
                last_frame_id = frame_id

        print(np.min(self.pos),np.max(self.pos))
        print(np.min(self.neg),np.max(self.neg))

        plt.figure(1)
        plt.hist(np.array(self.pos), bins=50, facecolor="green", edgecolor="green", alpha=0.7)

        plt.hist(np.array(self.neg), bins=50, facecolor="red", edgecolor="red", alpha=0.7)
        plt.savefig(os.path.join(self.save_fig_dir, "combine.png"))

        plt.figure(2)
        plt.hist(np.array(self.pos), bins=50, facecolor="green", edgecolor="green", alpha=0.7)
        plt.savefig(os.path.join(self.save_fig_dir, "pos.png"))
        plt.figure(3)
        plt.hist(np.array(self.neg), bins=50, facecolor="red", edgecolor="red", alpha=0.7)
        plt.savefig(os.path.join(self.save_fig_dir, "neg.png"))

    def get_person_bbox_list(self, image, info_dict, f_id):
        bboxes = []
        for pid in info_dict[f_id].keys():
            x, y, w, h = info_dict[f_id][pid]
            bboxes.append(np.array([x,y,x+w,y+h],dtype=np.float32))
        return bboxes


if __name__ == "__main__":
    #model = IOUModel()
    #model = GIOUModel()
    model = KeypointModel()
    eval_dir = "/home/wj/ai/mldata1/SportsMOT-2022-4-24/data/sportsmot_publish/dataset/val"
    #eval_dir = "/home/wj/ai/mldata/MOT/MOT16/train"
    save_fig = osp.join("/home/wj/ai/mldata1/SportsMOT-2022-4-24/tmp/geo_tools/",type(model).__name__)
    save_fig = wmlu.get_unused_path(save_fig)
    wmlu.create_empty_dir(save_fig,remove_if_exists=False)
    method = EvalGeoDis(model, eval_dir, save_fig_dir=save_fig)
    method.process()

import os
import cv2
import torch
import numpy as np
from distance import compute_distance_matrix
from fast_reid.onnx_infer_api import FastReidModel
from reid_yxd import reid_yxd_torch
from mfast_reids.fast_reid import MFastReIDS
from mfast_reid_t0.fast_reid import MFastReIDT0
from mfast_reid_t1.fast_reid import MFastReIDT1
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import wml_utils as wmlu
import os.path as osp
import sys

sys.path.append("..")

os.environ['CUDA_VISIBLE_DEVICES'] = "1"


class EvalReid:
    def __init__(self, model, eval_dir, save_fig_dir):
        self.reid_model = model
        self.eval_dir = eval_dir
        self.save_fig_dir = save_fig_dir
        self.pos = []
        self.neg = []
        self.type = "mean"

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
        person_list1 = self.get_person_list(image_1, gt_dict, frame_idx)

        person_feature1 = self.reid_model(np.array(person_list1))

        return person_feature1

    def process(self):
        dir_list = self.get_dir()
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
                if not torch.is_tensor(person_feature1):
                    person_feature1 = torch.from_numpy(person_feature1)
                if not torch.is_tensor(person_feature2):
                    person_feature2 = torch.from_numpy(person_feature2)

                distmat = compute_distance_matrix(person_feature1.to("cuda"),
                                                  person_feature2.to("cuda"),
                                                  metric="cosine")
                distmat = distmat.cpu().numpy()
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
                                    if i == idx2:
                                        continue
                                    self.neg.append(v)

                    else:
                        if self.type == "mean":
                            neg = (np.sum(distmat[idx1, :])) / (distmat.shape[1])
                            self.neg.append(neg)
                        else:
                            for i, v in enumerate(distmat[idx1, :]):
                                self.neg.append(v)
                last_features = person_feature2
                last_frame_id = frame_id

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

    def get_person_list(self, image, info_dict, f_id):
        image_list = []
        for pid in info_dict[f_id].keys():
            x, y, w, h = info_dict[f_id][pid]
            person = image[int(y):int(y + h), int(x):int(x + w), :]
            if person.shape[0]==0 or person.shape[1]==0:
                person = np.zeros([self.reid_model.size[1],self.reid_model.size[0],3],dtype=np.uint8)
            else:
                person = cv2.resize(person, self.reid_model.size)
            image_list.append(person[:, :, ::-1])
        return image_list


if __name__ == "__main__":
    #model = TransReIDSSL()
    #model = MFastReID()
    #model = CentroidReidModel()
    #model = MFastReID17()
    #model = reid_yxd_torch.ReidYXDModelTorch()
    #model = MFastReIDS()
    #model = FastReidModel()
    model = MFastReIDT1()
    eval_dir = "/home/wj/ai/mldata1/SportsMOT-2022-4-24/data/sportsmot_publish/dataset/val"
    #eval_dir = "/home/wj/ai/mldata/MOT/MOT16/train"
    save_fig = osp.join("/home/wj/ai/mldata1/SportsMOT-2022-4-24/tmp/reid_tools/",type(model).__name__)
    save_fig = wmlu.get_unused_path(save_fig)
    wmlu.create_empty_dir(save_fig,remove_if_exists=False)
    method = EvalReid(model, eval_dir, save_fig_dir=save_fig)
    method.process()

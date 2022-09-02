import os
import pickle

import cv2
import numpy as np

from postprocess_tools import vis_utils
from post_process import txt_to_dict, txt_to_dictv2, batch_frame_iou, get_query_feature, normal_feature
from postprocess_tools import recurse_get_filepath_in_dir, create_empty_dir
from postprocess_tools import visualize

if __name__ == "__main__":
    image_dir = r"/home/yangxiaodong/DataDisk_112/sportsmot_publish/dataset/val/"
    txt_dir = r"/home/yangxiaodong/DataDisk_112/sportsmot_publish/dataset/val_wj/data"
    save_dir = r"/home/yangxiaodong/Data/sportsmot/val"
    create_empty_dir(save_dir, remove_if_exists=True)
    for name in os.listdir(image_dir):
        txt_path = os.path.join(txt_dir, name + '.txt')
        image_path = os.path.join(image_dir, name)
        visualize.visualize(txt_path, image_path, save_dir=os.path.join(save_dir, name), show=False)
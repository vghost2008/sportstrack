import os
import cv2
import numpy as np

from postprocess_tools import vis_utils
from post_process import txt_to_dict
from postprocess_tools import recurse_get_filepath_in_dir, create_empty_dir


def visualize(txt_path, image_dir, save_dir=None, show=False):
    info_dict = txt_to_dict(txt_path)
    for keys in info_dict.keys():
        image_path = os.path.join(image_dir, "img1", str(int(keys)).zfill(6) + '.jpg')
        assert os.path.exists(image_path)
        image = cv2.imread(image_path)
        pid_dict = info_dict[keys]
        for pid in pid_dict.keys():
            bbox = pid_dict[pid]
            bbox = np.array(bbox).reshape(-1, ).astype(np.int)
            color = vis_utils.color_fn(pid)
            # cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color)
            vis_utils.draw_bbox(image, bbox, is_relative_bbox=False, label=pid, color=color, xywh=True,
                                xy_order=True)
        if save_dir:
            create_empty_dir(save_dir, remove_if_exists=False)
            cv2.imwrite(os.path.join(save_dir, str(keys).zfill(6) + '.jpg'), image)
        if show:
            cv2.imshow(str(keys), image)
            cv2.waitKey(0)
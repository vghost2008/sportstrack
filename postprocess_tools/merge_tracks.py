import os
import pickle
import sys
sys.path.append("..")
import shutil
import numpy as np
from post_process import txt_to_dict, txt_to_dictv2, batch_frame_iou, get_query_feature, normal_feature
import post_process
from postprocess_tools import create_empty_dir
from mfast_reid_t1.fast_reid import MFastReIDT1 as ReIDModel
import argparse
from dis_tools import distance

def parse_args():
    parser = argparse.ArgumentParser(description='MOT')
    parser.add_argument('--image_dir', type=str,default=r"/home/wj/ai/mldata1/SportsMOT-2022-4-24/data/sportsmot_publish/dataset/test", help='input img dir')
    parser.add_argument('--input_txt_dir', type=str,default=r"/home/wj/ai/mldata1/SportsMOT-2022-4-24/tmp/sportsmot-test/PDSMV2SportsTrackerT1/data", help='input dir path')
    parser.add_argument('--img_save_dir', type=str,default=r"/home/wj/ai/mldata1/SportsMOT-2022-4-24/tmp/postprocess_vis", help='output img dir, for debug')
    parser.add_argument('--output_txt_dir', type=str,default=r"/home/wj/ai/mldata1/SportsMOT-2022-4-24/tmp/postprocess_output",help='output txt dir')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    image_dir = args.image_dir
    txt_dir = args.input_txt_dir
    save_dir = args.img_save_dir
    txt_dir_save = args.output_txt_dir
    create_empty_dir(txt_dir_save, remove_if_exists=False)
    #reid_model = ReidYXDModelTorch()
    reid_model = ReIDModel()
    for name in os.listdir(image_dir):
        # if name !='v_00HRwkvvjtQ_c007':
        #     continue
        print(f"{name} is started")
        txt_path = os.path.join(txt_dir, name + '.txt')
        txt_save_path = os.path.join(txt_dir_save, name + '.txt')
        assert os.path.exists(txt_path)
        info_dict = txt_to_dict(txt_path)
        info_dict2 = txt_to_dictv2(txt_path)
        # p_frame_list = [list(info_dict2[pid].keys()) for pid in info_dict2.keys()]
        # dist_i = batch_frame_iou(p_frame_list, p_frame_list)
        pkl_path = os.path.join(txt_dir, name + '.pkl')
        if not os.path.exists(pkl_path):
            query_feature = get_query_feature(info_dict,info_dict2, os.path.join(image_dir, name), reid_model=reid_model)
            f_save = open(pkl_path, 'wb')
            pickle.dump(query_feature, f_save)
            f_save.close()
        else:
            with open(pkl_path, 'rb') as f:
                query_feature = pickle.load(f)
        pid_list = sorted([x for x in list(query_feature.keys()) if type(x) is int])
        features = []
        features_p = []
        features_n = []
        pid_p = []
        pid_n = []
        remove_id = []
        p_feature_dict = {}
        for pid in pid_list:
            feature = query_feature[pid]
            feature = np.array(feature)
            n_feature = normal_feature(feature)
            n_feature, feature = post_process.filter_feature(n_feature, feature)
            if post_process.eval_feature(n_feature, feature):
                if len(info_dict2[pid].keys()) < 10:
                    remove_id.append(pid)
                    continue
                p_feature_dict[pid] = {}
                p_feature_dict[pid]['n_feature'] = n_feature
                p_feature_dict[pid]['feature'] = feature
                features_p.append(n_feature)
                pid_p.append(pid)
            else:
                features_n.append(n_feature)
                pid_n.append(pid)
            # features.append(n_feature)
        p_frame_list = [list(info_dict2[pid].keys()) for pid in info_dict2.keys() if pid in pid_p]

        dist_i = batch_frame_iou(p_frame_list, p_frame_list)
        dist_i[dist_i > 0] = 1
        features_p = np.concatenate(features_p,axis=0)
        dist_f = distance.compute_distance_matrix(features_p, features_p, metric='cosine')
        dist = (dist_f + dist_i).numpy()
        dist = np.tril(dist)
        dist[dist == 0] = 1
        index = np.where(dist < 0.2)
        if len(index[0]) < 1:
            print(f"Min is {np.min(dist)},{dist.shape}")
            print(f"Copy file")
            shutil.copy(txt_path,txt_save_path)
            continue

        index_value = []
        for x, y in zip(index[0], index[1]):
            index_value.append((x, y, dist[x, y]))
        index = post_process.post_index(index_value)
        final_dict = post_process.process(index, p_feature_dict, pid_p, info_dict, info_dict2,
                                          os.path.join(image_dir, name), reid_model=reid_model)
        for pid in remove_id:
            remove_frame = info_dict2[pid]
            for fid in remove_frame:
                final_dict[fid].pop(pid)
                print(f"remove frame {fid} - {pid}")
        with open(txt_save_path, 'w') as f:
            for fid in final_dict.keys():
                for pid in final_dict[fid].keys():
                    x, y, w, h = final_dict[fid][pid]
                    f.write('{},{},{},{},{},{},{},{},{},{}\n'.format(fid, pid, x, y, w, h, -1, -1, -1, -1))
        print(f"{name} is done")
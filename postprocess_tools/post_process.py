import os.path
from dis_tools import distance
import cv2
import numpy as np
from tqdm import tqdm
import copy


def txt_to_dict(txt_path):
    with open(txt_path, encoding="utf-8") as f:
        content = f.readlines()
    info_dict = {}
    for line in content:
        f_id, p_id, xmin, ymin, w, h, _, _, _, _ = [float(x) for x in line.strip().split(",")]
        f_id = int(f_id)
        p_id = int(p_id)
        if f_id not in info_dict:
            info_dict[f_id] = {}
            info_dict[f_id][p_id] = [xmin, ymin, w, h]
        else:
            info_dict[f_id][p_id] = [xmin, ymin, w, h]
    return info_dict


def txt_to_dictv2(txt_path):
    with open(txt_path, encoding="utf-8") as f:
        content = f.readlines()
    info_dict = {}
    for line in content:
        f_id, p_id, xmin, ymin, w, h, _, _, _, _ = [float(x) for x in line.strip().split(",")]
        f_id = int(f_id)
        p_id = int(p_id)
        if p_id not in info_dict:
            info_dict[p_id] = {}
            info_dict[p_id][f_id] = [xmin, ymin, w, h]
        else:
            info_dict[p_id][f_id] = [xmin, ymin, w, h]
    return info_dict


def frame_iou(f_list1, f_list2):
    intersection = [i for i in f_list1 if i in f_list2]
    union = list(set(f_list1).union(set(f_list2)))
    if len(intersection) == 0:
        return 0
    else:
        return len(intersection) / len(union)


def batch_frame_iou(frame_list1, frame_list2):
    dist = np.empty((len(frame_list1), len(frame_list2)))
    for i, f_list1 in enumerate(frame_list1):
        for j, f_list2 in enumerate(frame_list2):
            dist[i][j] = frame_iou(f_list1, f_list2)
    return dist


def get_query_featureV2(info_dict, frame_list, pid, image_dir, reid_model=None):
    """
    info_dict[person_id][frame_id][bboxes]
    """
    features = {}
    query_image = []
    for fid in frame_list:
        image = cv2.imread(os.path.join(image_dir, "img1", str(fid).zfill(6) + '.jpg'))
        if pid in info_dict[fid]:
            bbox = info_dict[fid][pid]
            bbox = np.array(bbox).reshape(-1, ).astype(np.int)
            crop_image = image[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2], :]
            # cv2.imshow("crop_image", crop_image)
            crop_image = cv2.resize(crop_image, (128, 256))
            query_image.append(crop_image[:, :, ::-1])
            # cv2.imshow("crop_image_resize", crop_image)
            # cv2.waitKey(0)
    query_feature = get_query_reid(np.array(query_image), reid_model=reid_model)
    return query_feature


def get_query_feature(info_dict, image_dir, reid_model=None):
    """
    info_dict[person_id][frame_id][bboxes]
    """
    features = {}
    for pid in info_dict.keys():
        frame_ids = info_dict[pid].keys()
        query_image = []
        features[pid] = []
        features[str(pid) + "_frame_id"] = list(frame_ids)
        for frame_id in tqdm(frame_ids):
            image = cv2.imread(os.path.join(image_dir, "img1", str(frame_id).zfill(6) + '.jpg'))
            bbox = info_dict[pid][frame_id]
            bbox = np.array(bbox).reshape(-1, ).astype(np.int)
            crop_image = image[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2], :]
            # cv2.imshow("crop_image", crop_image)
            crop_image = cv2.resize(crop_image, reid_model.size)
            query_image.append(crop_image[:, :, ::-1])
            # cv2.imshow("crop_image_resize", crop_image)
            # cv2.waitKey(0)
        query_feature = get_query_reid(np.array(query_image), reid_model=reid_model)
        features[pid] = query_feature
    return features


def normal_feature(feature):
    feature = np.array(feature)
    feature_lens = len(feature)
    feature = np.sum(feature, axis=0)
    feature = feature / feature_lens
    feature = feature / np.sqrt(np.sum(feature ** 2))
    feature = feature.reshape(1, -1)
    return feature


def eval_feature(n_feature, features):
    diff = (distance.compute_distance_matrix(n_feature, features, metric="cosine")).numpy().reshape(-1, )
    diff_std = np.std(diff)
    return True
    # if diff_std < 0.08:
    #     return True
    # else:
    #     return False

def filter_feature(n_feature, features):
    diff = (distance.compute_distance_matrix(n_feature, features, metric="cosine")).numpy().reshape(-1, )
    features = features[np.where(diff < 0.2)]
    n_feature = normal_feature(features)
    return n_feature, features


def get_query_reid(query, reid_model=None, split_num=64):
    query_feature = reid_model(query)
    query_feature = np.array(query_feature)
    assert len(query_feature.shape) == 2, f"error shape {query_feature.shape}"
    query_feature = np.array(query_feature)
    return query_feature


# def post_dist(dist, pid_list, features_p, info_dict, info_dict2, image_dir, reid_model):
#     i1, i2 = np.where(dist == np.min(dist))[0]
#     p1, p2 = pid_list[i1], pid_list[i2]
#     p1_feature = features_p[i1]
#     p1_frame_list = [x for x in list(info_dict2[p1]) if x not in list(info_dict2[p2])]
#     p2_frame_list = [x for x in list(info_dict2[p2]) if x not in list(info_dict2[p1])]
#     get_query_featureV2(info_dict, frame_list=p2_frame_list, pid=p2, image_dir=image_dir, reid_model=reid_model)

def post_index(index_value):
    #index_value = np.array(index_value)
    #index_value = np.sort(index_value, axis=0)
    print(index_value)
    index_value.sort(key=lambda x:x[2])
    print(index_value)
    index = [index_value[0][0], index_value[0][1]]
    if len(index_value) == 1:
        index = np.array(index).reshape(-1, 2).astype(np.int)
        return index
    else:
        for i in range(1, len(index_value)):
            if index_value[i][0] in index or index_value[i][1] in index:
                continue
            else:
                index.append(index_value[i][0])
                index.append(index_value[i][1])

    index = np.array(index).reshape(-1, 2).astype(np.int)
    return index

def process(index, p_feature_dict, pid_list, info_dict, info_dict2, image_dir, reid_model=None):
    final_dict = copy.deepcopy(info_dict)
    for i1, i2 in zip(index[:, 0], index[:, 1]):
        # i1, i2 = np.where(dist == np.min(dist))[0]
        p1, p2 = pid_list[i1], pid_list[i2]
        if abs(p1 - p2) < 10:
            continue

        anchor_id = min(p1, p2)
        modify_id = max(p1, p2)
        modify_list = [x for x in list(info_dict2[modify_id]) if x not in list(info_dict2[anchor_id])]
        anchor_list = [x for x in list(info_dict2[anchor_id]) if x not in list(info_dict2[modify_id])]
        fea_dim =  np.array(p_feature_dict[anchor_id]['feature']).shape[-1]
        anchor_feature = np.array(p_feature_dict[anchor_id]['feature']).reshape(-1, fea_dim)
        # modify_feature = get_query_featureV2(info_dict, frame_list=modify_list, pid=modify_id,
        #                                      image_dir=image_dir,
        #                                      reid_model=reid_model)
        modify_feature = np.array(p_feature_dict[modify_id]['feature']).reshape(-1, fea_dim)
        # if len(modify_list) != 0:
        #     modify_feature = get_query_featureV2(info_dict, frame_list=modify_list, pid=modify_id,
        #                                          image_dir=image_dir,
        #                                          reid_model=reid_model)
            # p1_feature = np.array(features_p[anchor_id]).reshape(-1, 512)
            # p2_feature = np.array(features_p[modify_id]).reshape(-1, 512)
            # p1_frame_list = [x for x in list(info_dict2[anchor_id]) if x not in list(info_dict2[modify_id])]
            # p2_frame_list = [x for x in list(info_dict2[p2]) if x not in list(info_dict2[p1])]
            # if len(p1_frame_list) != 0 and len(p2_frame_list) != 0:
            #     if len(p1_frame_list) < len(p2_frame_list):
            #         p1_feature = get_query_featureV2(info_dict, frame_list=p1_frame_list, pid=p1,
            #                                          image_dir=image_dir,
            #                                          reid_model=reid_model)
            #         anchor_id = p2
            #         modify_id = p1
            #         modify_list = p1_frame_list
            #     else:
            #         p2_feature = get_query_featureV2(info_dict, frame_list=p2_frame_list, pid=p2,
            #                                          image_dir=image_dir,
            #                                          reid_model=reid_model)
            #         anchor_id = p1
            #         modify_id = p2
            #         modify_list = p2_frame_list
        dist_tmp = distance.compute_distance_matrix(anchor_feature, modify_feature, metric="cosine")
        dist_tmp = dist_tmp.numpy()
        if len(np.where(dist_tmp[:, 0] < 0.3)[0]) < (len(anchor_feature * len(modify_feature)) // 2):
            continue
        # modify_list = np.array(modify_list)[np.where(dist_tmp < 0.25)[0]]
        for i in modify_list:
            final_dict[i][anchor_id] = info_dict[i][modify_id]
            final_dict[i].pop(modify_id)
            print(f"frame {i}-{modify_id} is update {anchor_id}")
    return final_dict

from re import L
import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist
import object_detection2.bboxes as odb
import toolkit 
from cython_bbox import bbox_overlaps as bbox_ious
import object_detection2.wmath as wmath
import time

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious

def iousv2(atlbrs, btlbrs,a_covariances,b_covariances,covariances_scale=0.5):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious
    bboxes = np.transpose(btlbrs)
    res = []
    a_covariances = np.maximum(a_covariances.copy()*covariances_scale,1.0)
    b_covariances = np.maximum(b_covariances.copy()*covariances_scale,1.0)
    for i,bbox_ref in enumerate(atlbrs):
        bbox_ref = np.expand_dims(bbox_ref,axis=-1)
        int_ymin = np.maximum(bboxes[0], bbox_ref[0])
        int_xmin = np.maximum(bboxes[1], bbox_ref[1])
        int_ymax = np.minimum(bboxes[2], bbox_ref[2])
        int_xmax = np.minimum(bboxes[3], bbox_ref[3])
        h = np.maximum(int_ymax - int_ymin, 0.)
        w = np.maximum(int_xmax - int_xmin, 0.)
        inter_vol = h * w
        union_vol = -inter_vol \
                    + (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1]) \
                    + (bbox_ref[2] - bbox_ref[0]) * (bbox_ref[3] - bbox_ref[1])
        add_union_vol = a_covariances[i]*b_covariances
        union_vol = union_vol+add_union_vol
        jaccard = wmath.npsafe_divide(inter_vol, union_vol, 'jaccard')
        res.append(jaccard)
    
    return np.array(res,dtype=np.float32)


def gious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious
    
    for i in range(len(atlbrs)):
        bbox0 = atlbrs[i]
        _gious = odb.npgiou([bbox0],btlbrs)
        ious[i] = _gious

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def iou_distancev2(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    
    a_covariances = np.array([track.covariance_item for track in atracks],dtype=np.float32)
    b_covariances = np.array([track.covariance_item for track in btracks],dtype=np.float32)
    _ious = iousv2(atlbrs, btlbrs,a_covariances,b_covariances)
    cost_matrix = 1 - _ious

    return cost_matrix

def bboxes_iou(atlbrs, btlbrs):
    _ious = ious(atlbrs, btlbrs)

    return _ious

def bboxes_iouv2(atlbrs, btlbrs):
    ioua = []
    ioub = []
    for bbox in atlbrs:
        ioua.append(odb.npbboxes_intersection_of_box0(bbox,btlbrs))
    for bbox in btlbrs:
        ioub.append(odb.npbboxes_intersection_of_box0(bbox,atlbrs))
    
    ioua = np.array(ioua)
    ioub = np.stack(ioub,axis=-1)
    _ious = np.maximum(ioua,ioub)
    
    return _ious

def bboxes_iouv3(atlbr, btlbrs):
    ioua = odb.npbboxes_intersection_of_box0([atlbr],btlbrs)
    ioub = odb.npbboxes_intersection_of_box0(btlbrs,[atlbr])
    _ious = np.maximum(ioua,ioub)
    
    return _ious

def kps_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    bbox0 = [track.ltrb for track in atracks]
    bbox1 = [track.ltrb for track in btracks]
    kps0 = [track.cur_kps for track in atracks]
    kps1 = [track.cur_kps for track in btracks]

    return toolkit.kps_dis_matrix(kps0,kps1,bbox0,bbox1)

def kps_distancev2(atracks, btracks, threshold=0.1):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    dists = kps_distance(atracks,btracks)
    '''dists_a = kps_distance(atracks,atracks)+np.eye(len(atracks),dtype=np.float32)
    dists_b = kps_distance(btracks,btracks)+np.eye(len(btracks),dtype=np.float32)

    maska = dists_a<threshold
    maskb = dists_b<threshold

    idxs = np.where(maska)
    for x in idxs[0]:
        dists[x,:] = 1.0

    idxs = np.where(maskb)
    for x in idxs[0]:
        dists[:,x] = 1.0'''
    
    return dists

def giou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _gious = gious(atlbrs, btlbrs)
    cost_matrix = 1 - (_gious+1)/2

    return cost_matrix

def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost

def fuse_giou(cost_matrix, tracks, detections,threshold=0.75,out_data=None,alpha=0.9):
    if cost_matrix.size == 0:
        return cost_matrix
    giou_dist = giou_distance(tracks, detections)
    iou_dist = iou_distance(tracks,detections)

    max_cost = np.ones_like(cost_matrix)*2
    cost_matrix = np.where(giou_dist>threshold,max_cost,cost_matrix)
    scaled_matrix = cost_matrix*(giou_dist+1.0)
    cost_matrix = np.where(iou_dist>=0.99,scaled_matrix,cost_matrix)
    if out_data is not None:
        out_data.append(giou_dist)
        out_data.append(iou_dist)
    cost_matrix = cost_matrix*alpha+giou_dist*(1-alpha)
    return cost_matrix

def fuse_giouv2(cost_matrix, tracks, detections,threshold=0.75,out_data=None,alpha=0.9):
    if cost_matrix.size == 0:
        return cost_matrix
    giou_dist = giou_distance(tracks, detections)
    iou_dist = iou_distance(tracks,detections)

    max_cost = np.ones_like(cost_matrix)*2
    cost_matrix = np.where(giou_dist>threshold,max_cost,cost_matrix)
    scaled_matrix = cost_matrix*(giou_dist+1.0)
    cost_matrix = np.where(iou_dist>=0.99,scaled_matrix,cost_matrix)
    if out_data is not None:
        out_data.append(giou_dist)
        out_data.append(iou_dist)
    return cost_matrix

def fuse_giouv3(cost_matrix, tracks, detections,threshold=0.75):
    if cost_matrix.size == 0:
        return cost_matrix
    giou_dist = giou_distance(tracks, detections)

    max_cost = np.ones_like(cost_matrix)*2
    cost_matrix = np.where(giou_dist>threshold,max_cost,cost_matrix)
    return cost_matrix

def fuse_embedding(cost_matrix,atracks,btracks,threshold=0.35):
    e_cost = embedding_distance(atracks,btracks)
    mask = e_cost>threshold
    max_cost = np.ones_like(cost_matrix)
    for i,track in enumerate(atracks):
        if hasattr(track,"is_hard"):
            if track.is_hard:
                mask[i] = False
        if hasattr(track,"is_easy"):
            if track.is_easy:
                mask[i] = False

    cost_matrix = np.where(mask,max_cost,cost_matrix)

    return cost_matrix

def fuse_embeddingv2(cost_matrix,atracks=None,btracks=None,alpha=0.9,e_cost=None):
    if e_cost is None:
        e_cost = embedding_distance(atracks,btracks)
    cost_matrix = cost_matrix*alpha+e_cost*(1-alpha)

    return cost_matrix

def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost

def ho_distance(atracks,btracks,scales=[1.0,2.0,2.0],giou_stop=0.6,kps_stop=0.8,embedding_stop=0.3):
    '''
    scales: for giou,embedding
    '''
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    _gious = gious(atlbrs, btlbrs)
    giou_dis = 1 - (_gious+1)/2
    max_dis = np.ones_like(giou_dis)

    giou_mask = giou_dis>giou_stop
    giou_dis = np.where(giou_mask,max_dis,giou_dis)

    kps_dis = kps_distance(atracks,btracks)
    k_mask = kps_dis>kps_stop
    kps_dis = np.where(k_mask,max_dis,kps_dis)

    e_dis = embedding_distance(atracks,btracks)
    e_mask = e_dis>embedding_stop
    e_dis = np.where(e_mask,max_dis,e_dis)

    mask0 = np.logical_and(giou_mask,e_mask)
    mask1 = np.logical_and(giou_mask,k_mask)
    mask2 = np.logical_and(e_mask,k_mask)
    mask = np.logical_or(mask0,mask1)
    mask = np.logical_or(mask,mask2)

    giou_dis = giou_dis*scales[0]
    kps_dis = kps_dis*scales[1]
    e_dis = e_dis*scales[2]

    dis = np.minimum(giou_dis,e_dis)
    dis = np.minimum(dis,kps_dis)
    dis = np.where(mask,max_dis,dis)

    return dis
import numpy as np
import object_detection2.bboxes as odb
import math
from data_types import TrackState


def kps_dis(kp0,kp1,bbox0,bbox1,min_nr=4,inf_dis=1e8):
    score0 = kp0[:,-1]
    score1 = kp1[:,-1]
    mask0 = score0>0.5
    mask1 = score1>0.5

    base_mask = np.ones([17],dtype=np.bool)
    base_mask[[7,8,9,10,13,14,15,16]] = False
    mask = np.logical_and(mask0,mask1)
    #mask = np.logical_and(mask,base_mask)
    total_nr = np.count_nonzero(mask)

    wh0 = bbox0[2:]-bbox0[:2]
    wh1 = bbox1[2:]-bbox1[:2]
    S = np.min((wh0+wh1)/2)

    if total_nr < min_nr:
        dis = 1.0-odb.npbboxes_jaccard([bbox0],[bbox1])
        return dis[0]
    
    delta = (kp1[mask]-kp0[mask])[:,:2]
    delta = np.square(delta)
    delta = np.sum(delta,axis=-1)
    delta = np.sqrt(delta)
    mean_dis = np.mean(delta)

    dis = mean_dis/S
    dis = min(max(dis,0),1.0)

    return dis
    
def kps_dis_matrix(kps0,kps1,bboxes0,bboxes1):
    kps0 = np.array(kps0)
    kps1 = np.array(kps1)
    dis_matrix = np.ones([kps0.shape[0],kps1.shape[0]],dtype=np.float32)

    for i in range(kps0.shape[0]):
        for j in range(kps1.shape[0]):
            dis_matrix[i,j] = kps_dis(kps0[i],kps1[j],bboxes0[i],bboxes1[j])
    
    return dis_matrix

def kps_bboxes_nms(bboxes,kps,threshold=0.1,iou_threshold=0.2):
    if len(bboxes)<2:
        return bboxes,kps

    kps_matrix = kps_dis_matrix(kps,kps,bboxes,bboxes)
    nr_bboxes = len(bboxes)
    mask = np.ones([nr_bboxes],dtype=np.bool)

    for i in range(nr_bboxes-1):
        ious = odb.npbboxes_jaccard([bboxes[i]],bboxes)
        for j in range(i+1,nr_bboxes):
            if ious[j]<iou_threshold:
                continue
            if kps_matrix[i,j]<threshold:
                mask[j] = False
    
    mask = mask.tolist()
    return bboxes[mask],kps[mask],mask

def log_print(*args):
    #print(*args)
    pass

def remove_half_kps(bboxes,kps):
    if len(bboxes)<=1:
        return [True]
    kps = kps[:,[13,14,15,16],:]
    kps_score = kps[...,-1]
    keep = np.sum(np.array(kps_score>0.3).astype(np.int32),axis=-1)>1
    keep_bboxes = bboxes[keep.tolist()]
    for i in range(len(bboxes)):
        if keep[i]:
            continue
        bbox = bboxes[i]
        gious_dis = (1-odb.npgiou([bbox],keep_bboxes))/2
        if np.any(gious_dis<0.65):
            keep[i] = True

    return keep

def align_kps(kp0,kp1,threshold=0.4):
    base_mask = np.ones([17],dtype=np.bool)
    base_mask[[7,8,9,10,13,14,15,16]] = False
    mask0 = kp0[...,-1]>threshold
    mask1 = kp1[...,-1]>threshold
    mask = np.logical_and(np.logical_and(mask0,mask1),base_mask)
    if np.any(mask):
        idx = np.argmax(mask)
    else:
        mask = np.logical_and(mask0,mask1)
        if not np.any(mask):
            return None,None
        idx = np.argmax(mask)
    kp0[...,:2] = kp0[...,:2]-np.expand_dims(kp0[idx,:2],axis=0)
    kp1[...,:2] = kp1[...,:2]-np.expand_dims(kp1[idx,:2],axis=0)
    return kp0,kp1

def simple_kps_dis(kp0,kp1,threshold=0.4):
    mask0 = kp0[...,-1]>threshold
    mask1 = kp1[...,-1]>threshold
    mask = np.logical_and(mask0,mask1)
    mask = mask.tolist()
    if not np.any(mask):
        return -1
    dis = (kp0[mask]-kp1[mask])[...,:2]
    dis = np.square(dis)
    dis = np.sum(dis,axis=-1)
    dis = np.mean(dis)
    return math.sqrt(dis)

def simple_aligned_kps_dis(kp0,kp1,threshold=0.4):
    kp0,kp1 = align_kps(kp0,kp1,threshold=threshold)
    if kp0 is None:
        return -1
    
    return simple_kps_dis(kp0,kp1,threshold=threshold)


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

    return kps_dis_matrix(kps0,kps1,bbox0,bbox1)

def set_untracked_dis(dis_matrix,tracks,v=1.0):
    if dis_matrix.shape[0] != len(tracks):
        print(f"ERROR size {dis_matrix.shape[0]} vs {len(tracks)}")
        return dis_matrix
    
    res = dis_matrix.copy()

    for i,track in enumerate(tracks):
        if(track.state == TrackState.Lost):
            res[i] = v
    
    return res

def gather_tracks_by_idx(tracks_pool,idxs):
    '''
    idx in idxs may not in stracks_pool
    '''
    res = []
    for track in tracks_pool:
        if track.track_idx in idxs:
            res.append(track)
    
    return res

if __name__ == "__main__":
    kps0 = np.array([[      487.1,      464.06,     0.45359],
       [      487.1,      458.19,     0.53741],
       [     483.19,       462.1,     0.46846],
       [      487.1,      458.19,     0.61229],
       [     475.35,       462.1,     0.60248],
       [     491.02,      454.27,     0.54884],
       [      473.4,      475.81,      0.3269],
       [     494.94,      436.65,     0.49745],
       [     469.48,      487.56,     0.31506],
       [     504.73,      420.98,     0.43875],
       [     475.35,      516.94,     0.48977],
       [     518.44,      513.02,      0.5471],
       [     502.77,      511.06,     0.63261],
       [     555.65,      514.98,     0.63434],
       [      520.4,      497.35,     0.80151],
       [     586.98,      524.77,     0.66719],
       [     549.77,      520.85,     0.44264]], dtype=np.float32)
    kps1 = np.array([[     497.93,      417.93,     0.35145],
       [     495.71,       414.6,     0.38472],
       [     495.71,      416.82,     0.29749],
       [     496.82,      411.27,     0.54249],
       [     502.37,      404.62,     0.25975],
       [     501.26,      409.05,     0.59756],
       [     522.34,      400.18,      0.4783],
       [     512.35,      433.46,     0.43057],
       [     547.85,      404.62,     0.51891],
       [     510.13,       462.3,     0.37314],
       [     562.27,      410.16,     0.47261],
       [      530.1,      427.91,     0.62482],
       [     548.96,      423.48,     0.65236],
       [     521.23,      457.87,     0.71844],
       [     537.87,      457.87,      0.8253],
       [     523.45,       497.8,     0.70386],
       [      542.3,      493.37,     0.64326]], dtype=np.float32)
    bbox0 = np.array([     461.47,       412.7,      617.43,      568.98], dtype=np.float32)
    bbox1 = np.array([     491.34,      385.63,      557.83,      503.51], dtype=np.float32)
    print(odb.npbboxes_jaccard([bbox0],[bbox1]))
    dis = kps_dis_matrix(np.array([kps0]),np.array([kps1]),[bbox0],[bbox1])
    print(dis)
    #print(simple_aligned_kps_dis(kps0,kps1))

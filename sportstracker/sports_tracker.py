import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F
import object_detection2.bboxes as odb
import math
from .kalman_filter import KalmanFilter
import matching
from .basetrack import BaseTrack
from data_types import TrackState, Angle
from .strack import STrack 
from toolkit import *
from bot_sort_tracker.gmc import GMC
import object_detection2.visualization as odv
import img_utils as wmli


class SportsTracker(object):
    try_match_nr = 0
    pre_match_nr = 0
    embedding_match_nr = 0
    kps_match_nr = 0
    iou_match_nr = 0
    second_kps_match_nr = 0
    second_iou_match_nr = 0
    
    def __init__(self, det_thresh=0.2,frame_rate=30,track_buffer=30,assignment_thresh=[0.8,0.5,0.7,0.7],nms_thresh=0.5,dir_path=None):
        print(f"pid={os.getpid()} track_buffer={track_buffer}")
        print(f"pid={os.getpid()} det threshold {det_thresh}, frame_rate={frame_rate}, track_buffer={track_buffer}, thresh={assignment_thresh}, nms_threshold={nms_thresh}")
        Angle.delta = 90
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = det_thresh
        self.track_thresh = det_thresh-0.1
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.img_size = None #(H,W)
        self.assignment_thresh = assignment_thresh
        self.pred_bboxes = None
        self.last_bboxes = None
        self.pred_track_id = None
        self.need_kps = True
        self.cur_kps = None
        self.cur_frame = None
        self.keypoints = None
        self.nms_thresh = nms_thresh
        self.dir_path = dir_path
        self.gmc = GMC(method="file",gmc_file=osp.join(dir_path,"gmc.txt"))
        self.track_id_trans_dict = {}

    def __del__(self):
        print(f"pid={os.getpid()}, pre_match_nr={self.pre_match_nr}, embedding_match_nr={self.embedding_match_nr}, kps_match_nr={self.kps_match_nr}, iou_match_nr={self.iou_match_nr}, second_kps_match_nr={self.second_kps_match_nr}, second_iou_match_nr={self.second_iou_match_nr}")


    def apply(self,bboxes,probs,ids,is_first_frame=False):
        img_info = (100,100)
        img_size = (100,100)
        if is_first_frame:
            BaseTrack._count = 0
            self.frame_id = 0
        bboxes = odb.npchangexyorder(bboxes)
        return self.update(bboxes,probs,ids,img_info,img_size)

    def update(self, bboxes,probs, ids, img_info, img_size):
        '''
        probs: 要求为降序
        bboxes: [N,4] (x0,y0,x1,y1)
        '''
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        matched_dets = []

        scores = probs
        bboxes = bboxes

        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        bboxes_nr = scores.shape[0]
        all_indexs = np.array(list(range(bboxes_nr)),dtype=np.int32)
        inds_remain = scores >= self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        inds_remain = np.where(inds_remain)[0].tolist()
        inds_second = np.where(inds_second)[0].tolist()


        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            track.set_track_idx(-1)
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        STrack.multi_predict(tracked_stracks)
        lost_pred_time = 10
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame<= lost_pred_time:
                track.predict()

        self.lost_stracks,tracked_stracks = self.try_merge_tracks(self.lost_stracks,tracked_stracks)
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        warp = self.gmc.apply(self.cur_frame, bboxes)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        for strack in strack_pool:
            if strack.state != TrackState.Lost:
                continue
            if self.get_lost_bbox_pos(strack.raw_tlbr) is not None and \
                self.get_lost_bbox_pos(strack.tlbr) is None:
                strack.restore()

        for x in strack_pool:
            x.tmp_pred = np.array(x.yminxminymaxxmax).copy()

        self.pred_kps(strack_pool)
        strack_pool_nr = len(strack_pool)
        for x in strack_pool:
            x.set_track_idx(-1)

        all_detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s,id) for
                        (tlbr, s,id) in zip(bboxes, scores,ids)]
        for i,det in enumerate(all_detections):
            det.set_track_idx(i)
            det.set_kps(self.cur_kps[i])

        if len(self.lost_stracks)>1:
            log_print("A")
        
        self.mark_hard_easy_tracks(strack_pool)
        self.set_fake_new_track(strack_pool,all_detections,self.cur_frame.shape,threshold=0.5)

        #######################################try match
        dists = matching.embedding_distance(strack_pool, all_detections)
        raw_e_dis = dists.copy()
        dists = matching.fuse_giouv3(dists,strack_pool,all_detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, 
                                        thresh=self.assignment_thresh[3])
        
        self.try_match_nr += len(matches)
        if len(matches)>3:
            have_embedding_match = True
        else:
            have_embedding_match = False
        #######################################try match end

        dists = matching.kps_distancev2(strack_pool, all_detections)
        '''e_dists = matching.embedding_distance(strack_pool, all_detections)*2.5
        e_dists = matching.fuse_giouv2(e_dists, strack_pool, all_detections)
        dists = np.where(dists<e_dists,dists,e_dists)'''
        dists = matching.fuse_embeddingv2(dists,strack_pool,all_detections)
        dists = set_untracked_dis(dists,strack_pool)

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.05)

        log_print(f"pre matches {len(matches)}/{len(all_indexs)},{strack_pool_nr}")
        self.pre_match_nr += len(matches)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = all_detections[idet]
            log_print(track.track_id,dists[itracked,idet])
            if track.state == TrackState.Tracked:
                track.update(all_detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            matched_dets.append(det.raw_tlbr)

        #iou process
        all_detections = [all_detections[i] for i in u_detection]
        detections = gather_tracks_by_idx(all_detections,inds_remain)
        detections_second = gather_tracks_by_idx(all_detections,inds_second)

        strack_pool = [strack_pool[i] for i in u_track]
        ######################################################################


        # Predict the current location with KF

        dists = matching.embedding_distance(strack_pool, detections)
        #dists = matching.iou_distance(strack_pool, detections)
        tmp_outdata = []
        #log_print([x.track_id for x in strack_pool])
        dists = matching.fuse_giou(dists, strack_pool, detections,out_data=tmp_outdata)
        matches, u_track, u_detection = matching.linear_assignment(dists, 
                                        thresh=self.assignment_thresh[3])
        log_print(f"{self.frame_id} ------------------------------------------")
        log_print(f"embedding match {len(matches)}/{len(all_indexs)},{strack_pool_nr}")
        self.embedding_match_nr += len(matches)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            matched_dets.append(det.raw_tlbr)
            log_print(track.track_id,dists[itracked,idet],[x[itracked,idet] for x in tmp_outdata])
        
        detections = [detections[i] for i in u_detection]
        #strack_pool = [strack_pool[i] for i in u_track]
        __strack_pool = [strack_pool[i] for i in u_track]
        strack_pool = []
        for strack in __strack_pool:
            if strack.state == TrackState.Lost and strack.lost_pose is not None:
                lost_stracks.append(strack)
            else:
                strack_pool.append(strack)

        ####
        dists = matching.kps_distancev2(strack_pool, detections)
        if have_embedding_match:
            dists = matching.fuse_embeddingv2(dists,strack_pool,detections)
        #jif not self.args.mot20:
          #  dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.assignment_thresh[0])

        log_print(f"kps matches {len(matches)}/{len(all_indexs)},{strack_pool_nr}")
        self.kps_match_nr += len(matches)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            log_print(track.track_id,dists[itracked,idet])
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            matched_dets.append(det.raw_tlbr)

        #iou process
        detections = [detections[i] for i in u_detection]
        strack_pool = [strack_pool[i] for i in u_track]

        ####
        dists = matching.iou_distancev2(strack_pool, detections)
        if have_embedding_match:
            dists = matching.fuse_embeddingv2(dists,strack_pool,detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.assignment_thresh[0])

        log_print(f"iou matches {len(matches)}/{len(all_indexs)},{strack_pool_nr}")
        self.iou_match_nr += len(matches)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            log_print(track.track_id,dists[itracked,idet])
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            matched_dets.append(det.raw_tlbr)
        #Step 4: Second association, with low score detection boxes
        # association the untrack to the low score detections
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        #dists = matching.iou_distance(r_tracked_stracks, detections_second)
        dists = matching.kps_distancev2(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=self.assignment_thresh[1])

        log_print(f"kps second matches {len(matches)}/{len(all_indexs)},{strack_pool_nr}")
        self.second_kps_match_nr += len(matches)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            log_print(track.track_id,dists[itracked,idet])
        
        #iou process
        detections_second = [detections_second[i] for i in u_detection_second]
        r_tracked_stracks = [r_tracked_stracks[i] for i in u_track]

        ####
        dists = matching.iou_distancev2(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=self.assignment_thresh[1])

        log_print(f"iou second matches {len(matches)}/{len(all_indexs)},{strack_pool_nr}")
        self.second_iou_match_nr += len(matches)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            log_print(track.track_id,dists[itracked,idet])


        for it in u_track:
            track = r_tracked_stracks[it]
            if track.fake_new_track is not None:
                bbox = track.fake_new_track.raw_tlbr.copy()
                if track.update_with_fake_new_track(self.frame_id):
                    matched_dets.append(bbox)
                    activated_starcks.append(track)
                    continue
            if not track.state == TrackState.Lost:
                lost_pose = self.get_lost_bbox_pos(track.raw_tlbr)
                track.mark_lost(lost_pose=lost_pose)
                lost_stracks.append(track)

        #Deal with unconfirmed tracks, usually tracks with only one beginning frame
        detections = [detections[i] for i in u_detection]
        #dists = matching.iou_distance(unconfirmed, detections)
        dists = matching.kps_distancev2(unconfirmed, detections)
        #jif not self.args.mot20:
        #    dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=self.assignment_thresh[2])
        log_print(f"kps unconfirmed")
        for itracked, idet in matches:
            det = detections[idet]
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
            log_print(unconfirmed[itracked].track_id,dists[itracked,idet])
            matched_dets.append(det.raw_tlbr)

        #process iou
        detections = [detections[i] for i in u_detection]
        unconfirmed = [unconfirmed[i] for i in u_unconfirmed]
        #dists = matching.iou_distance(unconfirmed, detections)
        dists = matching.iou_distancev2(unconfirmed, detections)
        #jif not self.args.mot20:
        #    dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=self.assignment_thresh[2])
        log_print(f"iou unconfirmed")
        for itracked, idet in matches:
            det = detections[idet]
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
            log_print(unconfirmed[itracked].track_id,dists[itracked,idet])
            matched_dets.append(det.raw_tlbr)

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        #Step 4: Init new stracks
        matched_dets = np.array(matched_dets)
        detections_bboxes = [detections[i].raw_tlbr for i in u_detection]
        _,keep = odb.npbbxoes_nms(detections_bboxes,self.nms_thresh)
        for i,inew in enumerate(u_detection):
            if not keep[i]:
                continue
            track = detections[inew]
            bbox = track.raw_tlbr
            if len(matched_dets)>0 and np.any(matching.bboxes_iouv3(bbox,matched_dets)>self.nms_thresh):
                continue
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        #Step 5: Update state
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # log_print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        #self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks,msg=self.dir_path)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        '''self.pred_bboxes = np.array([x.tmp_pred for x in self.tracked_stracks if hasattr(x,"tmp_pred")])
        self.last_bboxes = np.array([x.raw_yminxminymaxxmax for x in self.tracked_stracks if hasattr(x,"tmp_pred")])
        self.pred_track_id = np.array([x.track_id for x in self.tracked_stracks if hasattr(x,"tmp_pred")],dtype=np.int32)'''

        track_ids = []
        track_bboxes = []
        track_idxs = []
        for track in output_stracks:
            track_ids.append(track.track_id)
            track_bboxes.append(track.yminxminymaxxmax)
            track_idxs.append(track.track_idx)
        return np.array(track_ids),np.array(track_bboxes),np.array(track_idxs)

    def pred_kps(self,tracks):
        bboxes = [track.ltrb for track in tracks]
        if len(bboxes)>0:
            kps = self.keypoints.get_kps_by_bboxes(self.cur_frame,bboxes,scale_bboxes=True)
            for i,track in enumerate(tracks):
                track.cur_kps = kps[i]
                #track.set_kps(kps[i])
        else:
            kps = []
        
        return kps

    def save_kps(self,path,kps):
        frame = odv.draw_keypoints(self.cur_frame.copy(),kps,r=2,line_thickness=1)
        wmli.imwrite(path,frame)

    def save_track(self,path,track):
        frame = odv.draw_keypoints(self.cur_frame.copy(),[track.cur_kps],r=2,line_thickness=1)
        frame = odv.draw_bboxes_xy(frame,[track.track_id],bboxes=[track.tlbr],
                            is_relative_coordinate=False,show_text=True,
                            thickness=2,
                            font_scale=1.0)
        wmli.imwrite(path,frame)

    @staticmethod
    def mark_hard_easy_tracks(tracks,threshold_hard=0.55,threshold_easy=0.9):
        if len(tracks)==0:
            return
        dists = matching.iou_distance(tracks,tracks)
        dists = dists+np.eye(len(tracks),dtype=np.float32)
        mask = dists<threshold_hard
        idxs = np.where(mask)
        idxs = idxs[0].tolist()
        for i,track in enumerate(tracks):
            if i in idxs:
                track.is_hard = True
            else:
                track.is_hard = False

        mdists = np.min(dists,axis=1)
        mask = mdists>threshold_easy
        idxs = np.where(mask)
        idxs = idxs[0].tolist()
        for i,track in enumerate(tracks):
            if i in idxs:
                track.is_easy = True
            else:
                track.is_easy = False

    def set_fake_new_track(self,tracks,detections,img_size,threshold=0.5):
        '''
        img_size:[H,W]
        '''
        if len(tracks)==0:
            return
        dists = matching.iou_distance(tracks,detections)
        idxs = np.argmin(dists,axis=1)
        min_value = np.min(dists,axis=1)
        for i,track in enumerate(tracks):
            track.fake_new_track = None
            if min_value[i]>threshold or track.state == TrackState.Lost or len(track.f_history)<5:
                continue
            if track.is_hard:
                track_bbox = track.tlwh
                bbox = [track_bbox[2],track_bbox[3],img_size[1]-track_bbox[2],img_size[0]-track_bbox[3]]
                if odb.is_point_in_bbox(track_bbox[:2],bbox):
                    track.fake_new_track = detections[idxs[i]]
    
    def try_merge_tracks(self,losted_tracks,tracks,new_track_frame_nr=10,embedding_threshold=None,match_nr=3):

        if len(losted_tracks) == 0 or len(tracks)==0:
            return losted_tracks,tracks

        if embedding_threshold is None:
            embedding_threshold = self.assignment_thresh[3]
        img_size = self.cur_frame.shape[:2]
        new_losted_tracks = []
        new_tracks = []
        dists_matrix = np.ones([len(losted_tracks),len(tracks)],dtype=np.float32)
        for i,ltrack in enumerate(losted_tracks):
            if ltrack.track_len()<new_track_frame_nr:
                continue
            p0 = self.get_lost_bbox_pos(ltrack.raw_tlbr)
            if p0 is None:
                new_losted_tracks.append(ltrack)
                continue
            for j,track in enumerate(tracks):
                if track.track_len()<3:
                    continue
                if ltrack.end_frame>=track.start_frame or track.end_frame-track.start_frame>new_track_frame_nr:
                    continue
                p1 = self.get_lost_bbox_pos(track.tlwh_to_tlbr(track.first_tlwh))
                if p1 is None:
                    continue
                if p0 != p1:
                    continue
                #dists = ltrack.features_dis(ltrack.sf_history,[track.f_history[-1]])
                dists = ltrack.features_dis(ltrack.sf_history,[track.f_history[-1]])
                dists = np.reshape(dists,[-1])
                mask = dists<embedding_threshold
                nr = np.count_nonzero(mask)
                if nr>= match_nr:
                    v = np.sum(dists[mask])/math.pow(nr,1.1)
                    dists_matrix[i,j] = v
                    #ltrack.merge(track,msg=self.dir_path)
                    #track.state = TrackState.Removed
        while True:
            x,y = np.unravel_index(np.argmin(dists_matrix),dists_matrix.shape)
            v = dists_matrix[x,y]
            if v>embedding_threshold:
                break
            ltrack = losted_tracks[x]
            track = tracks[y]
            ltrack.merge(track,msg=self.dir_path)
            if track.track_id in self.track_id_trans_dict:
                print(f"ERROR: id {track.track_id} already in dict, frame id = {self.frame_id}, msg={self.dir_path}")
            self.track_id_trans_dict[track.track_id] = ltrack.track_id
            track.state = TrackState.Removed
            dists_matrix[x] = 1.0
            dists_matrix[:,y] = 1.0

        for ltrack in losted_tracks:
            if ltrack.state != TrackState.Lost:
                new_tracks.append(ltrack)
            else:
                new_losted_tracks.append(ltrack)

        for track in tracks:
            if track.state != TrackState.Removed:
                new_tracks.append(track)

        return new_losted_tracks,new_tracks


    def get_lost_bbox_pos(self,bbox):
        '''

        Args:
            bbox: [x0,y0,x1,y1]

        Returns:

        '''
        img_size = self.cur_frame.shape[:2]
        size_delta = 10
        img_bbox = [size_delta,size_delta,img_size[1]-size_delta,img_size[0]-size_delta]

        if bbox[2]>=img_bbox[2] or bbox[0]<=img_bbox[0] or bbox[1]<=img_bbox[1] or bbox[3]>=img_bbox[3]:
            cx = (bbox[0]+bbox[2])/2
            cy = (bbox[1]+bbox[3])/2
            img_cx = img_size[1]/2
            img_cy = img_size[0]/2
            dx = cx-img_cx
            dy = cy-img_cy
            return Angle(x=dx,y=dy)
        return None
    
    def post_process(self,track_datas):
        print(type(self).__name__,"post process")
        if len(track_datas)==0:
            return track_datas
        if len(self.track_id_trans_dict) != 0:
            new_track_datas = []
            for idx,tid,cur_bbox in track_datas:
                if tid in self.track_id_trans_dict:
                    tid = self.track_id_trans_dict[tid]
                new_track_datas.append([idx,tid,cur_bbox])
            track_datas = new_track_datas

        track_ids = []
        for idx,tid,*_ in track_datas:
            track_ids.append(tid)
        track_ids = np.array(track_ids,dtype=np.int32)
        save_nr_threshold = 5
        new_track_datas = []
        id2keep = set()
        id2remove = set()
        for idx,tid,cur_bbox in track_datas:
            if tid in id2remove:
                continue
            if tid in id2keep:
                new_track_datas.append([idx,tid,cur_bbox])
                continue
            nr = np.count_nonzero(track_ids==tid)
            if nr<save_nr_threshold:
                id2remove.add(tid)
            else:
                new_track_datas.append([idx,tid,cur_bbox])
                id2keep.add(tid)
        track_datas = new_track_datas

        return track_datas

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb,msg=""):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
            print(f"Remove {p} by duplicate with {q}, msg={msg}")
        else:
            dupa.append(p)
            print(f"Remove {q} by duplicate with {p}, msg={msg}")
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

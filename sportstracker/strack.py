import numpy as np
from .basetrack import BaseTrack
from data_types import TrackState
from .kalman_filter import KalmanFilter
from collections import deque
import object_detection2.keypoints as odk
import object_detection2.bboxes as odb
from scipy.spatial.distance import cdist


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    UPDATE_FEATURE_THRESHOLD = 0.3
    BUFFER_SIZE = 30
    MAX_FAKE_TRACK_NR = 5

    def __init__(self, tlwh, score, feat=None, track_idx=None,feat_history=50):
        '''
        tlwh: (x0,y0,w,h)
        '''

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.first_tlwh = np.asarray(tlwh,dtype=np.float)
        self.score = score
        self.tracklet_len = 0
        self.track_idx = track_idx
        self.smooth_feat = None
        self.curr_feat = None
        self.f_history = deque([], maxlen=feat_history)
        self.sf_history = deque([], maxlen=feat_history)
        self.bbox_history = deque([], maxlen=feat_history)
        self.frame_id_history = deque([], maxlen=feat_history)
        self.bbox_history.append(self._tlwh.copy())
        if feat is not None:
            self.update_features(feat)
        self.alpha = 0.9
        self.cur_kps = None
        self.fake_match_nr = 0
        self.fake_new_track = None

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.frame_id - self.start_frame>self.BUFFER_SIZE and len(self.f_history)>10:
            dis = 1.0-np.sum(feat*self.smooth_feat)
            if dis>self.UPDATE_FEATURE_THRESHOLD:
                #print(self.frame_id,self,1.0-np.sum(feat*self.smooth_feat))
                return
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.f_history.append(feat.copy())
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)
        self.sf_history.append(self.smooth_feat.copy())
        self.frame_id_history.append(self.frame_id)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov
    
    def store(self):
        self.backup_mean = self.mean
        self.backup_covariance = self.covariance

    def restore(self):
        self.mean = self.backup_mean
        self.covariance = self.backup_covariance

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].store()
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.fake_match_nr = 0
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def merge(self,new_track,msg=""):
        print(f"Merge {self} {new_track}, msg={msg}")
        self.mean = new_track.mean.copy()
        self.covariance = new_track.covariance.copy()
        self._tlwh = new_track._tlwh
        self.bbox_history = new_track.bbox_history
        self.update_features(new_track.f_history[-1])
        self.fake_match_nr = 0
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.update_features(new_track.curr_feat)
        self.frame_id = new_track.end_frame
        self.score = new_track.score

    def re_activate(self, new_track, frame_id, new_id=False):

        self.fake_match_nr = 0
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        self.frame_id = frame_id
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self._tlwh = new_track._tlwh
        self.bbox_history.append(self._tlwh.copy())
        self.set_track_idx(new_track.track_idx)
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.fake_match_nr = 0
        self.frame_id = frame_id
        self.tracklet_len += 1

        self._tlwh = new_track._tlwh
        self.bbox_history.append(self._tlwh.copy())

        new_tlwh = new_track.tlwh

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True
        self.set_track_idx(new_track.track_idx)

        self.score = new_track.score
    
    def update_with_fake_new_track(self, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        new_track = self.fake_new_track
        if new_track is None:
            return False
        
        if self.fake_match_nr>=self.MAX_FAKE_TRACK_NR:
            return False
            
        self.frame_id = frame_id

        print("UPDATE with fake target:",self.frame_id,self)

        self.fake_match_nr += 1
        self.tracklet_len += 1

        self._tlwh = new_track._tlwh
        self.bbox_history.append(self._tlwh.copy())

        new_tlwh = new_track.tlwh

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True
        self.set_track_idx(new_track.track_idx)

        self.score = new_track.score
        self.fake_new_track = None

        return True

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        '''
        (x0,y0,x1,y1) -> (x0,y0,w,h)
        '''
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def yminxminymaxxmax(self):
        ret = self.tlwh.copy()
        return np.array([ret[1],ret[0],ret[1]+ret[3],ret[0]+ret[2]],dtype=np.float32)

    def __repr__(self):
        return 'OT_{}_{}({}-{})'.format(self.track_id, self.track_idx,self.start_frame, self.end_frame)
    
    def set_track_idx(self,idx):
        self.track_idx = idx
    
    @property
    def raw_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self._tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def raw_yminxminymaxxmax(self):
        ret = self._tlwh.copy()
        return np.array([ret[1],ret[0],ret[1]+ret[3],ret[0]+ret[2]],dtype=np.float32)
    
    @property
    def ltrb(self):
        '''
        由于历史原因，与tlbr返回一致
        '''
        return self.tlbr
    
    @property
    def covariance_item(self):
        if self.covariance is None:
            return (self.tlwh[-1]+self.tlwh[-2])/40.0
        return np.mean([self.covariance[0,0],self.covariance[1,1],self.covariance[2,2],self.covariance[3,3]])

    def set_kps(self,kps):
       self.cur_kps = kps 
       bbox = odk.npget_bbox(kps,0.1)
       if bbox is None:
           return
       cur_bbox = self.ltrb
       delta = 10
       iou = odb.npbboxes_jaccard([bbox],[cur_bbox])[0]
       if iou<0.1:
           print(f"ERROR in set kps: {kps}, {bbox}, {cur_bbox} {self.start_frame}")

    @staticmethod
    def features_dis(featuresa,featuresb=None):
        if featuresb is None:
            same_fea = True
            featuresb = featuresa
        else:
            same_fea = False

        cost_matrix = np.zeros((len(featuresa), len(featuresa)), dtype=np.float)
        if cost_matrix.size == 0:
            return cost_matrix
        featuresa = np.asarray(featuresa, dtype=np.float)
        featuresb = np.asarray(featuresb, dtype=np.float)
        cost_matrix = np.maximum(0.0, cdist(featuresa, featuresb, "cosine"))  # Nomalized features
        if same_fea:
            cost_matrix = np.tril(cost_matrix,k=0)
        return cost_matrix

    def set_track_idx(self,idx):
        self.track_idx = idx

    def track_len(self):
        return self.end_frame-self.start_frame





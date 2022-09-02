import numpy as np
import cv2
from itertools import count
from demo_toolkit import *
import object_detection2.visualization as odv
import object_detection2.bboxes as odb
import img_utils as wmli
from collections import Iterable
import time
import copy
from byte_tracker.byte_tracker import BYTETracker
from sportstracker.sports_tracker import SportsTracker
from bot_sort_tracker.bot_sort import BoTSORT
import random
import wml_utils as wmlu
import pickle
from keypoints.get_keypoints import KPDetection
from toolkit import kps_bboxes_nms,remove_half_kps


colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

def get_text_pos_fn0(pmin,pmax,bbox,label):
    p1 = (pmax[0],pmin[1])
    return (p1[0]-5,p1[1])

def get_text_pos_fn1(pmin,pmax,bbox,label):
    p1 = ((pmin[0]+pmax[0])//2,pmin[1])
    return p1
def get_text_pos_fn2(pmin,pmax,bbox,label):
    p1 = (pmax[0],pmax[1])
    return (p1[0]-5,p1[1])

def random_get_text_pos_fn(pmin,pmax,bbox,label):
    funs = [get_text_pos_fn0,get_text_pos_fn1,get_text_pos_fn2]
    return random.choice(funs)(pmin,pmax,bbox,label)

def text_fnv0(label,score):
    return f"{label}_{score:.2f}"

def color_fn(label):
    color_nr = len(colors_tableau)
    return colors_tableau[label%color_nr]

def npscale_bboxes(bboxes,scale,correct=False,max_size=None):
    if not isinstance(scale,Iterable):
        scale = [scale,scale]
    ymin,xmin,ymax,xmax = bboxes[...,0],bboxes[...,1],bboxes[...,2],bboxes[...,3]
    cy = (ymin+ymax)/2.
    cx = (xmin+xmax)/2.
    h = ymax-ymin
    w = xmax-xmin
    h = scale[0]*h
    w = scale[1]*w
    ymin = cy - h / 2.
    ymax = cy + h / 2.
    xmin = cx - w / 2.
    xmax = cx + w / 2.
    xmin = np.maximum(xmin,0)
    ymin = np.maximum(ymin,0)
    if max_size is not None:
        xmax = np.minimum(xmax,max_size[1]-1)
        ymax = np.minimum(ymax,max_size[0]-1)
    data = np.stack([ymin, xmin, ymax, xmax], axis=-1)
    return data

def cut_and_resizev0(img,bboxes,size=(288,384)):
    res = []
    bboxes = np.array(bboxes).astype(np.int32)
    bboxes = np.maximum(bboxes,0)
    bboxes[...,0::2] = np.minimum(bboxes[...,0::2],img.shape[1])
    bboxes[...,1::2] = np.minimum(bboxes[...,1::2],img.shape[0])
    for bbox in bboxes:
        cur_img = img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
        if size is not None:
            if cur_img.shape[0]>1 and cur_img.shape[1]>1:
                cur_img = cv2.resize(cur_img,size,interpolation=cv2.INTER_LINEAR)
            else:
                cur_img = np.zeros([size[1],size[0],3],dtype=np.float32)
        res.append(cur_img)
    return np.array(res)

class TrackDemo:
    def __init__(self,save_path=None,max_frame_cn=None,config={},name=None,args=None,imgs_save_path=None,video_path=None):
        if "max_person_nr" not in config:
            self.max_person_nr = 0
        else:
            self.max_person_nr = config["max_person_nr"]
        print(f"Max person nr:",self.max_person_nr)
        self.config = config
        self.max_frame_cn = max_frame_cn
        self.args = args

        if self.config['model'] == "BYTETracker":
            det_thresh = config.get("det_thresh",0.7)
            self.tracker = BYTETracker(track_buffer=10,assignment_thresh=[0.9,0.7,0.3],det_thresh=det_thresh)
        elif self.config['model'] == "SportsTracker":
            det_thresh = config.get("det_thresh",0.7)
            reid_thresh = config.get("reid_thresh",0.4)
            track_buffer = config.get("track_buffer",90)
            thresh = config.get("thresh",[0.8,0.8,0.7])
            nms_thresh = config.get("nms_thresh",0.5)
            assignment_thresh=thresh+[reid_thresh]
            self.tracker = SportsTracker(track_buffer=track_buffer,
                                               assignment_thresh=assignment_thresh,
                                               dir_path=video_path,
                                               det_thresh=det_thresh,
                                               nms_thresh=nms_thresh)
        elif self.config['model'] == "BoTSORT":
            seq = name
            track_buffer = 30
            track_high_thresh = 0.6
            new_track_thresh = track_high_thresh + 0.1
            reid_thresh = config.get("reid_thresh",0.25)
            self.tracker = BoTSORT(track_buffer=track_buffer,
                                   det_thresh=new_track_thresh,
                                   dir_path=video_path,
                                   appearance_thresh=reid_thresh)

        self.imgs_save_path = imgs_save_path+"_finaly"
        self.person_det = config["person_det"]()
        det_model_name = type(self.person_det).__name__
        data_root = os.environ.get("SPORTSTRACK_ROOT","/home/wj/ai/mldata1/SportsMOT-2022-4-24")
        self.det_cache_dir = osp.join(data_root,"tmp/cache",det_model_name)
        wmlu.create_empty_dir(self.det_cache_dir,remove_if_exists=False)
        self.need_kps = self.need_kps(self.tracker)
        self.name = name
        det_cache_save_path = self.get_person_det_save_path()
        if osp.exists(det_cache_save_path):
            with open(det_cache_save_path,"rb") as f:
                print(f"Load cached person detection {det_cache_save_path}.")
                self.det_datas = pickle.load(f)
        else:
            self.det_datas = {}

        kps_det_cache_save_path = self.get_kps_det_save_path()
        if osp.exists(kps_det_cache_save_path):
            with open(kps_det_cache_save_path,"rb") as f:
                print(f"Load cached kps detection {kps_det_cache_save_path}.")
                self.kps_det_datas = pickle.load(f)
        else:
            self.kps_det_datas = {}

        self.tracked_datas = {}
        if self.config['use_reid']:
            self.image_encoder = self.config["reid_model"]()
        else:
            self.image_encoder = None
        self.force_save_kps = False
        self.idx = 0
        save_dir = osp.dirname(save_path)
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        self.save_path = save_path
        self.track_datas = []
        if self.need_kps:
            self.keypoints = KPDetection()
            self.tracker.keypoints = self.keypoints

    def get_person_det_save_path(self):
        return osp.join(self.det_cache_dir,self.name+".pkl")

    def get_kps_det_save_path(self):
        return osp.join(self.det_cache_dir,self.name+"_kps.pkl")

    def __del__(self):
        save_path = self.get_person_det_save_path()
        if not osp.exists(save_path) and len(self.det_datas)>20:
            with open(save_path,"wb") as f:
                pickle.dump(self.det_datas,f)
        save_path = self.get_kps_det_save_path()
        if (not osp.exists(save_path) and len(self.kps_det_datas)>20) or self.force_save_kps:
            with open(save_path,"wb") as f:
                pickle.dump(self.kps_det_datas,f)
        if len(self.track_datas):
            if hasattr(self.tracker,"post_process"):
                self.track_datas = self.tracker.post_process(self.track_datas)
                pass
            with open(self.save_path,"w") as f:
                for idx,tid,cur_bbox in self.track_datas:
                    data = f"{idx},{tid},{cur_bbox[0]},{cur_bbox[1]},{cur_bbox[2]-cur_bbox[0]},{cur_bbox[3]-cur_bbox[1]},-1,-1,-1,-1\n"
                    f.write(data)
            self.track_datas = []
        pass

    def track(self,frame):
        '''
        frame: rgb [H,W,3]
        '''
        det_save_path = self.get_person_det_save_path()

        if not osp.exists(det_save_path) or len(self.det_datas)<5:
            det_frame = frame[...,::-1]
            bboxes,probs = self.person_det(det_frame)
            self.det_datas[self.idx] = bboxes,probs
        else:
            if self.idx in self.det_datas:
                bboxes,probs = self.det_datas[self.idx]
            else:
                bboxes = []
                probs = []
        
        if self.need_kps:
            kps_det_save_path = self.get_kps_det_save_path()
            if not osp.exists(kps_det_save_path) or len(self.kps_det_datas)<5:
                kps = self.keypoints.get_kps_by_bboxes(frame,bboxes,scale_bboxes=True)
                self.kps_det_datas[self.idx] = kps
            else:
                if self.idx in self.kps_det_datas:
                    kps = self.kps_det_datas[self.idx]
                    if len(kps)!=len(bboxes):
                        kps = self.keypoints.get_kps_by_bboxes(frame,bboxes,scale_bboxes=True)
                        self.kps_det_datas[self.idx] = kps
                        self.force_save_kps = True
                else:
                    kps = np.zeros([0,5],dtype=np.float32)
            '''if self.config.get("kps_nms",True):
                bboxes,kps,mask = kps_bboxes_nms(bboxes,kps)
                probs = probs[mask]'''

            '''if len(bboxes)>0:
                keep = remove_half_kps(bboxes,kps)
                bboxes = bboxes[keep]
                kps = kps[keep]
                probs = probs[keep]'''

            self.tracker.cur_kps = kps
        self.tracker.cur_frame = np.array(frame).copy()

        if len(bboxes)<3:
            print(f"{self.name} idx={self.idx}, bboxes nr = {len(bboxes)}")

        if len(bboxes)==0:
            self.idx += 1
            return frame

        if self.max_person_nr>0 and probs.shape[0]>self.max_person_nr:
            bboxes = bboxes[:self.max_person_nr]
            probs = probs[:self.max_person_nr]

        org_bboxes = bboxes.copy()
        org_probs = probs.copy()
        '''
        tracker input/output bboxes format [y0,x0,y1,x1]
        '''
        if self.config["use_reid"]:
            embds_imgs = cut_and_resizev0(frame,bboxes.astype(np.int32),size=self.image_encoder.size)
            embds = self.image_encoder(embds_imgs)
            bboxes = odb.npchangexyorder(bboxes)
            tracked_id, tracked_bboxes, tracked_idx = self.tracker.apply(bboxes, probs,embds,
                                                                     is_first_frame=self.idx==0)
        else:
            bboxes = odb.npchangexyorder(bboxes)
            tracked_id, tracked_bboxes, tracked_idx = self.tracker.apply(bboxes, probs,
                                                                     is_first_frame=self.idx==0)
        tracked_id = tracked_id
        tracked_bboxes = tracked_bboxes
        self.idx += 1

        saved_bboxes = []
        if len(tracked_bboxes)>0:
            save_tracked_bboxes = odb.npchangexyorder(tracked_bboxes)
            save_bboxes = odb.npchangexyorder(bboxes)
            for tid,tbbox,tidx in zip(tracked_id,save_tracked_bboxes,tracked_idx):
                if tidx>=0:
                    cur_bbox = save_bboxes[tidx]
                else:
                    cur_bbox = tbbox
                saved_bboxes.append(cur_bbox.copy())
                #data = f"{self.idx},{tid},{cur_bbox[0]},{cur_bbox[1]},{cur_bbox[2]-cur_bbox[0]},{cur_bbox[3]-cur_bbox[1]},-1,-1,-1,-1\n"
                #self.track_datas.append([self.idx,tid,data,np.array(cur_bbox).copy()])
                self.track_datas.append([self.idx,tid,np.array(cur_bbox).copy()])

        frame = np.ascontiguousarray(frame)
        if self.args is None or self.args.log_imgs:
            frame = odv.draw_bboxes_xy(frame,tracked_id,bboxes=saved_bboxes,
                            color_fn=color_fn,
                            is_relative_coordinate=False,show_text=True,
                            thickness=2,
                            font_scale=1.0)
            '''if self.need_kps:
                frame = odv.draw_keypoints(frame,kps,r=2,line_thickness=1)
            frame = odv.draw_bboxes(frame, np.array(list(range(len(bboxes)))), bboxes=bboxes,
                                    color_fn=lambda x: (255, 255, 255),
                                    is_relative_coordinate=False, show_text=True,
                                    thickness=1,
                                    font_scale=0.6,
                                    get_text_pos_fn=get_text_pos_fn2,
                                    text_color=(255, 255, 255))'''
            '''bboxes = odb.npchangexyorder(org_bboxes)
            frame = odv.draw_bboxes(frame, np.array(list(range(len(bboxes)))), bboxes=bboxes,
                                    scores=org_probs,
                                    color_fn=lambda x: (255, 255, 255),
                                    is_relative_coordinate=False, show_text=True,
                                    thickness=1,
                                    font_scale=0.6,
                                    get_text_pos_fn=random_get_text_pos_fn,
                                    text_fn=text_fnv0,
                                    text_color=(255, 255, 255))'''
            if hasattr(self.tracker,"pred_bboxes") and self.tracker.pred_bboxes is not None:
                frame = odv.draw_bboxes(frame,np.array(list(range(len(bboxes)))),bboxes=bboxes,
                            color_fn=lambda x:(255,255,255),
                            is_relative_coordinate=False,show_text=True,
                            thickness=1,
                            font_scale=0.6,
                            get_text_pos_fn=get_text_pos_fn2,
                            text_color=(255,255,255))
                bboxes = self.tracker.pred_bboxes
                bboxes0 = self.tracker.last_bboxes
                labels = self.tracker.pred_track_id
                frame = odv.draw_bboxes(frame,labels,bboxes=bboxes,
                            color_fn=lambda x:(110,110,110),
                            is_relative_coordinate=False,show_text=True,
                            thickness=1,
                            font_scale=0.8,
                            get_text_pos_fn=get_text_pos_fn2,
                            text_color=(110,110,110))
                frame = odv.draw_bboxes(frame,labels,bboxes=bboxes0,
                            color_fn=lambda x:(110,0,0),
                            is_relative_coordinate=False,show_text=True,
                            thickness=1,
                            font_scale=0.6,
                            get_text_pos_fn=get_text_pos_fn1,
                            text_color=(110,0,0))

        return frame

    @staticmethod
    def need_kps(tracker):
        if hasattr(tracker,"need_kps") and tracker.need_kps:
            return True
        return False

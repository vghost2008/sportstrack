import cv2
import numpy as np
import os
import sys
import time
from vis_utils import *
import img_utils as wmli
import os.path as osp

__version__ = "1.2.3"

joints_pair = [[0 , 1], [1 , 2], [2 , 0], [1 , 3], [2 , 4], [3 , 5], [4 , 6], [5 , 6], [5 , 11],
[6 , 12], [11 , 12], [5 , 7], [7 , 9], [6 , 8], [8 , 10], [11 , 13], [13 , 15], [12 , 14], [14 , 16]]
colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
'''
boxes:[N,4],[ymin,xmin,ymax,xmax]
return:[ymin,xmin,ymax,xmax]
'''
def bbox_of_boxes(boxes):
    if not isinstance(boxes,np.ndarray):
        boxes = np.array(boxes)
    boxes = np.transpose(boxes)
    ymin = np.min(boxes[0])
    xmin = np.min(boxes[1])
    ymax = np.max(boxes[2])
    xmax = np.max(boxes[3])
    return np.array([ymin,xmin,ymax,xmax])

'''
boxes:[...,4] ymin,xmin,ymax,xmax
scale:[hscale,wscale]
'''
def npclip_bboxes(bboxes,max_size):
    ymin,xmin,ymax,xmax = bboxes[...,0],bboxes[...,1],bboxes[...,2],bboxes[...,3]
    xmin = np.maximum(xmin,0)
    ymin = np.maximum(ymin,0)
    if max_size is not None:
        xmax = np.minimum(xmax,max_size[1]-1)
        ymax = np.minimum(ymax,max_size[0]-1)
    data = np.stack([ymin, xmin, ymax, xmax], axis=-1)
    return data

class CycleBuffer:
    def __init__(self,cap=5):
        self.cap = cap
        self.buffer = []
    def append(self,v):
        self.buffer.append(v)
        l = len(self.buffer)
        if l>self.cap:
            self.buffer = self.buffer[l-self.cap:]

    def __getitem__(self, slice):
        return self.buffer[slice]

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer = []

class TimeThis():
    def __init__(self,name="TimeThis",auto_show=True):
        self.begin_time = 0.
        self.end_time = 0
        self.name = name
        self.auto_show = auto_show

    def __enter__(self):
        self.begin_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        if self.auto_show:
            te = (self.end_time-self.begin_time)*1000
            fps = 1000/(te+1e-8)
            print(f"{self.name}: total time {te:.3f}, FPS={fps:.3f}.")

    def time(self):
        return self.end_time-self.begin_time

def add_jointsv2(image, joints, color, r=5,threshold=0.01):

    def link(a, b, color):
        jointa = joints[a]
        jointb = joints[b]
        if jointa[2] > threshold and jointb[2] > threshold:
            cv2.line(
                image,
                (int(jointa[0]), int(jointa[1])),
                (int(jointb[0]), int(jointb[1])),
                color, 2 )

    # add link
    for pair in joints_pair:
        link(pair[0], pair[1], color)

    # add joints
    for i, joint in enumerate(joints):
        if joint[2] > threshold and joint[0] > 1 and joint[1] > 1:
            cv2.circle(image, (int(joint[0]), int(joint[1])), r, colors_tableau[i], -1)

    return image

def show_keypoints(image, joints, color=[0,255,0],threshold=0.01):
    image = np.ascontiguousarray(image)
    if color is None:
        use_random_color=True
    else:
        use_random_color = False
    for person in joints:
        if use_random_color:
            color = np.random.randint(0, 255, size=3)
            color = [int(i) for i in color]
        add_jointsv2(image, person, color=color,threshold=threshold)

    return image

class BufferTextPainter:
    def __init__(self,buffer_len=30):
        self.buffer_nr = buffer_len
        self.cur_text = ""
        self.cur_idx = 0
        self.font = os.path.join(os.path.dirname(__file__),"simhei.ttf")

    def putText(self,img,text,font_scale=20,text_color=(255,255,255)):
        img = np.ascontiguousarray(img)
        if text == "":
            if self.cur_idx == 0:
                return img
        else:
            self.cur_idx =self.buffer_nr
            self.cur_text = text

        self.cur_idx = self.cur_idx-1
        img = draw_text(img,(12,12),self.cur_text,
                        text_color=text_color,
                        font_size=font_scale,
                        font=self.font)
        return img

def resize_height(img,h,interpolation=cv2.INTER_LINEAR):
    shape = img.shape
    new_h = h
    new_w = int(shape[1]*new_h/shape[0])
    return cv2.resize(img,dsize=(new_w,new_h),interpolation=interpolation)

def resize_width(img,w,interpolation=cv2.INTER_LINEAR):
    shape = img.shape
    new_w = w
    new_h = int(shape[0]*new_w/shape[1])
    return cv2.resize(img,dsize=(new_w,new_h),interpolation=interpolation)

def expand_resize(img,size,interpolation=cv2.INTER_LINEAR):
    '''
    size: [w,h]
    '''
    r = max(size[0]/img.shape[1],size[1]/img.shape[0])
    new_h = int(r*img.shape[0]+0.5)
    new_w = int(r*img.shape[1]+0.5)
    return cv2.resize(img,dsize=(new_w,new_h),interpolation=interpolation)

def resize_short_size(img,size,interpolation=cv2.INTER_LINEAR):
    shape = img.shape
    if shape[0]<shape[1]:
        return resize_height(img,size,interpolation)
    else:
        return resize_width(img,size,interpolation)

class VideoDemo:
    def __init__(self,model,fps=30,save_path=None,buffer_size=0,show_video=True,max_frame_cn=None,interval=None,
         file_pattern="{:06d}.jpg",args=None) -> None:
        self.model = model
        self.fps = fps
        self.save_path = save_path
        self.buffer_size = buffer_size
        self.buffer = []
        self.write_size = None
        self.video_reader = None
        self.video_writer = None
        self.show_video = show_video
        self.preprocess = None
        self.max_frame_cn = max_frame_cn
        self.interval = interval
        self.file_pattern = file_pattern
        self.args = args
        print(f"Demo toolkit version {__version__}.")
        self.track_data = []
    
    def __del__(self):
        self.close()
    
    def close(self):
        if self.video_writer is not None:
            self.video_writer.release()
        if hasattr(self,"model"):
            del self.model

    def init_reader(self):
        if self.video_path is not None and os.path.exists(self.video_path):
            print(f"Use video file {self.video_path}")
            self.video_reader = wmli.VideoReader(self.video_path,file_pattern=self.file_pattern)
            self.frame_cnt = self.video_reader.frames_nr
            if self.max_frame_cn is not None and self.max_frame_cn>1:
                self.frame_cnt = min(self.frame_cnt,self.max_frame_cn)
        else:
            if self.video_path is not None:
                vc = int(self.video_path)
            else:
                vc = -1
            print(f"Use camera {vc}")
            self.video_reader = cv2.VideoCapture(vc)
            self.frame_cnt = -1


    def inference_loop(self,video_path=None):
        self.video_path = video_path
        self.init_reader()
        idx = 0

        for frame in self.video_reader:
            idx += 1
            if self.interval is not None and self.interval>1:
                if idx%self.interval != 0:
                    continue
            self.model.idx = idx
            if self.preprocess is not None:
                frame = self.preprocess(frame)
            img = self.inference(frame)
            save_path = osp.join(self.save_path,self.file_pattern.format(idx))
            if self.args is None or self.args.log_imgs:
                wmli.imwrite(save_path,img)
            if self.video_writer is not None:
                self.video_writer.write(img[..., ::-1])
            if self.show_video:
                cv2.imshow("video",img[...,::-1])
                if cv2.waitKey(30)&0xFF == 27:
                    break
            if self.frame_cnt > 1:
                #sys.stdout.write(f"{idx}/{self.frame_cnt}  {idx*100/self.frame_cnt:.3f}%.\r")
                if idx>self.frame_cnt:
                    break

    def inference(self,img):
        if self.buffer_size <= 1:
            r_img = self.inference_single_img(img)
        else:
            r_img = self.inference_buffer_img(img)
        return r_img

    def inference_single_img(self,img):
        return self.model(img)

    def inference_buffer_img(self,img):
        self.buffer.append(img)
        if len(self.buffer)>self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
        return self.model(self.buffer)

    @staticmethod
    def get_last_img(imgs):
        img = imgs[-1]
        if isinstance(img,dict):
            if 'raw_image' in img:
                return img['raw_image']
            return img['image']
        else:
            return img

    @staticmethod
    def resize_h_and_save_raw_image_preprocess(img,h=224):
        r_img = resize_height(img,h).astype(np.uint8)
        return {'image':r_img,"raw_image":img}

class IntervalMode:
    def __init__(self,interval=30):
        self.interval = interval
        self.idx = 0

    def add(self):
        self.idx += 1

    def need_pred(self):
        self.add()
        return (self.idx%self.interval)==0


def get_video_indexs(size,nr):
    delta = (size-1)/nr
    idxs = (np.arange(nr).astype(np.float32)*delta+delta/2).astype(np.int32)
    return idxs

def crop_imgs(imgs, crop_bbox):
    x1, y1, x2, y2 = crop_bbox
    return [img[y1:y2, x1:x2] for img in imgs]

def center_crop(imgs,crop_size):
    img_h, img_w = imgs[0].shape[:2]
    crop_w, crop_h = crop_size

    left = (img_w - crop_w) // 2
    top = (img_h - crop_h) // 2
    right = left + crop_w
    bottom = top + crop_h

    crop_bbox = np.array([left, top, right, bottom])

    return crop_imgs(imgs, crop_bbox)

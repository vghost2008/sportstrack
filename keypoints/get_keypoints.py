from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import Iterable
import os.path as osp
import cv2
import numpy as np
from .yolox import *
import cv2
import wml_utils as wmlu
from demo_toolkit import TimeThis

def npscale_bboxes(bboxes,scale,correct=False,max_size=None):
    if not isinstance(scale,Iterable):
        scale = [scale,scale]
    bboxes = np.array(bboxes)
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

curdir_path = osp.dirname(__file__)

class PersonDetection:
    def __init__(self):
        self.model = YOLOXDetection()

    def __call__(self, img):
        '''
        img: BGR order
        '''
        assert len(img.shape)==3,"Error img size"
        img_shape = img.shape
        output = self.model(img)
        mask = output[...,-1]==0
        output = output[mask]
        bboxes = output[...,:4]
        #labels = output[...,-1]
        probs = output[...,4]*output[...,5]

        wh = bboxes[...,2:]-bboxes[...,:2]
        bboxes = npscale_bboxes(bboxes,1.2,max_size=[img.shape[1],img.shape[0]])
        wh_mask = wh>10
        size_mask = np.logical_and(wh_mask[...,0],wh_mask[...,1])
        bboxes = bboxes[size_mask]
        probs = probs[size_mask]

        return bboxes,probs
    

def to_tensor(image):
    image = image.astype(np.float32)/255
    image = np.transpose(image,[2,0,1])
    return image

def npnormalize(x,mean,std):
    scale = np.reshape(np.array(std, dtype=np.float32), [3, 1, 1])
    offset = np.reshape(np.array(mean, dtype=np.float32), [3, 1, 1])
    x = (x-offset)/scale
    return x

class KPDetection:
    SIZE = (288,384)
    UPSCALE=6
    #SIZE = (192,256)
    def __init__(self) -> None:
        self.init_model()
        #self.person_det = PersonDetection()
        self.person_det = None
        self.device = torch.device("cuda:0")
        self.person_bboxes = None

    def init_model(self):
        onnx_path = osp.join(wmlu.parent_dir_path_of_file(__file__),"models","keypoints.torch")
        print(f"Load {onnx_path}")
        self.model = torch.jit.load(onnx_path)

    @staticmethod
    def preprocess(img):
        #img = to_tensor(img)
        #img = npnormalize(img,mean=[0.485, 0.456, 0.406],
        #                         std=[0.229, 0.224, 0.225])
        img = img.astype(np.float32)
        img = np.transpose(img,[2,0,1])
        return img


    @staticmethod
    def cut_and_resize(img,bboxes,size=(288,384)):
        res = []
        res_bboxes = []
        for i,bbox in enumerate(bboxes):
            cur_img = img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
            if cur_img.shape[0]>1 and cur_img.shape[1]>1:
                #cur_img = cv2.resize(cur_img,size,interpolation=cv2.INTER_LINEAR)
                cur_img,bbox = KPDetection.resize_img(cur_img,bbox,size)
            else:
                cur_img = np.zeros([size[1],size[0],3],dtype=np.float32)
            #cv2.imwrite(f"{i}.jpg",cur_img[...,::-1])
            res.append(cur_img)
            res_bboxes.append(bbox)
        return res,np.array(res_bboxes,dtype=np.float32)
    
    @staticmethod
    def resize_img(img,bbox,target_size,pad_color=(127,127,127)):
        res = np.ndarray([target_size[1],target_size[0],3],dtype=np.uint8)
        res[:,:] = np.array(pad_color,dtype=np.uint8)
        ratio = target_size[0]/target_size[1]
        bbox_cx = (bbox[2]+bbox[0])/2
        bbox_cy = (bbox[3]+bbox[1])/2
        bbox_w = (bbox[2]-bbox[0])
        bbox_h = (bbox[3]-bbox[1])
        if img.shape[1]>ratio*img.shape[0]:
            nw = target_size[0]
            nh = int(target_size[0]*img.shape[0]/img.shape[1])
            bbox_h = bbox_w/ratio
        else:
            nh = target_size[1]
            nw = int(target_size[1]*img.shape[1]/img.shape[0])
            bbox_w = bbox_h*ratio
    
        try:
            if img.shape[0]==0 or img.shape[1]==0:
                img = np.ones([nh,nw,3],dtype=np.uint8)
            else:
                img = cv2.resize(img,(nw,nh),interpolation=cv2.INTER_LINEAR)

            xoffset = (target_size[0]-nw)//2
            yoffset = (target_size[1]-nh)//2
            res[yoffset:yoffset+nh,xoffset:xoffset+nw] = img
            bbox = np.array([bbox_cx-bbox_w/2,bbox_cy-bbox_h/2,bbox_cx+bbox_w/2,bbox_cy+bbox_h/2],dtype=np.float32)
        except Exception as e:
            print(f"ERROR {e} {img.shape} {(nw,nh)}")
            pass
        return res,bbox


    @staticmethod
    def get_offset_and_scalar(bboxes,size):
        offset = bboxes[...,:2]
        offset = np.expand_dims(offset,axis=1)
        bboxes_size = bboxes[...,2:]-bboxes[...,:2]
        cur_size = np.array(size,np.float32)
        cur_size = np.resize(cur_size,[1,2])
        scalar = bboxes_size/cur_size
        scalar = np.expand_dims(scalar,axis=1)*KPDetection.UPSCALE
        return offset,scalar

    @staticmethod
    def get_max_preds(batch_heatmaps):
        '''
        get predictions from score maps
        heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
        '''
        assert isinstance(batch_heatmaps, np.ndarray), \
            'batch_heatmaps should be numpy.ndarray'
        assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'
        batch_size = batch_heatmaps.shape[0]
        num_joints = batch_heatmaps.shape[1]
        width = batch_heatmaps.shape[3]
        heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width #x
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width) #y

        pred_mask = np.tile(np.greater(maxvals, 0.05), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask
        return preds, maxvals

    @staticmethod
    def get_final_preds(batch_heatmaps):
        coords, maxvals = KPDetection.get_max_preds(batch_heatmaps)

        heatmap_height = batch_heatmaps.shape[2]
        heatmap_width = batch_heatmaps.shape[3]

        # post-processing
        if True:
            for n in range(coords.shape[0]):
                for p in range(coords.shape[1]):
                    hm = batch_heatmaps[n][p]
                    px = int(math.floor(coords[n][p][0] + 0.5))
                    py = int(math.floor(coords[n][p][1] + 0.5))
                    if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                        diff = np.array(
                            [
                                hm[py][px + 1] - hm[py][px - 1],
                                hm[py + 1][px] - hm[py - 1][px]
                            ]
                        )
                        coords[n][p] += np.sign(diff) * .25

        preds = coords.copy()

        #return preds, maxvals
        return np.concatenate([preds,maxvals],axis=-1)

    def get_person_bboxes(self,img,return_ext_info=False):
        '''

        Args:
            img: RGB order

        Returns:
            ans: [N,17,3] (x,y,score,...)
        '''
        img = img[...,::-1]
        bboxes,probs = self.person_det(img)
        if len(probs) == 0:
            return np.zeros([0,4],dtype=np.float32)
        return bboxes

    def get_kps_by_bboxes(self,img,bboxes,scale_bboxes=False):
        '''

        Args:
            img: RGB order

        Returns:
            ans: [N,17,3] (x,y,score,...)
        '''
        #print(bboxes)
        #cv2.imwrite("/home/wj/ai/mldata/0day/x1/a.jpg",img)
        if len(bboxes)==0:
            print(f"ERROR: empty bboxes.")
            return np.zeros([0,17,3],dtype=np.float32)

        if scale_bboxes:
            bboxes = npscale_bboxes(bboxes,1.2,max_size=[img.shape[1],img.shape[0]])
        imgs,bboxes = self.cut_and_resize(img,bboxes.astype(np.int32),KPDetection.SIZE)
        bboxes = np.array(bboxes)
        imgs = [self.preprocess(x) for x in imgs]
        imgs = np.ascontiguousarray(np.array(imgs))
        imgs = torch.Tensor(imgs).to(self.device)
        #torch.cuda.empty_cache()
        try:
            with torch.no_grad():
                output = self.model(imgs)
            output = output.detach().cpu().numpy()
        except Exception as e:
            print(f"ERROR:{e}")
            return np.zeros([imgs.shape[0],17,3],dtype=np.float32)
        output = self.get_final_preds(output)
        offset,scalar = self.get_offset_and_scalar(bboxes,KPDetection.SIZE)
        output[...,:2] = output[...,:2]*scalar+offset
        return output

    def process_fea_maps(self,output,org_bboxes,t_bboxes):
        batch_size = output.shape[0]
        
        for i in range(batch_size):
            org_bboxes_t = org_bboxes[i]
            t_bboxes_t = t_bboxes[i]
            w = t_bboxes_t[2]-t_bboxes_t[0]+1e-8
            h = t_bboxes_t[3]-t_bboxes_t[1]+1e-8
            x_beg = int((org_bboxes_t[0]-t_bboxes_t[0])*output.shape[3]/w)
            output[i,:,:,:x_beg] = 0.0
            x_end = int((org_bboxes_t[2]-t_bboxes_t[0])*output.shape[3]/w)
            output[i,:,:,x_end:] = 0.0
            y_beg = int((org_bboxes_t[1]-t_bboxes_t[1])*output.shape[2]/h)
            output[i,:,:y_beg,:] = 0.0
            y_end = int((org_bboxes_t[3]-t_bboxes_t[1])*output.shape[2]/h)
            output[i,:,y_end:,:] = 0.0
        
        return output



    def __call__(self, img):
        '''

        Args:
            img: RGB order

        Returns:
            ans: [N,17,3] (x,y,score,...)
        '''
        bboxes = self.get_person_bboxes(img,return_ext_info=False)
        self.person_bboxes = np.array(bboxes)[:12]
        return self.get_kps_by_bboxes(img,self.person_bboxes)

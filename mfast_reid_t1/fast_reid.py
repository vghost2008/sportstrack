import os.path as osp
import cv2
import os
import math
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torch.multiprocessing
import numpy as np
import onnxruntime as ort
import wtorch.utils as wtu
import wml_utils as  wmlu


curdir_path = osp.dirname(__file__)


class MFastReIDT1:
    def __init__(self) -> None:
        onnx_path = osp.join(wmlu.parent_dir_path_of_file(__file__), "models","fast_reid.torch")
        self.device = torch.device("cuda:0")
        print(f"Load {onnx_path}")
        self.model = torch.jit.load(onnx_path).to(self.device)
        self.size =(128,384)
    
    @wtu.split_forward_batch32
    def __call__(self, imgs):
        '''
        RGB order [B,H,W,C]
        '''
        with torch.no_grad():
            imgs = np.array(imgs,dtype=np.float32)
            imgs = torch.tensor(imgs)
            imgs = imgs.permute(0,3,1,2)
            imgs = imgs.cuda()
            ids = self.model(imgs).cpu().numpy()
        return ids



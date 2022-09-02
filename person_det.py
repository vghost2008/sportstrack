import numpy as np
import wml_utils as wmlu
import img_utils as wmli
import torch
from persondetsmv2.yolox import YOLOXDetectionS as PersonDetSMSV2

class PersonDetectionSMSV2:
    def __init__(self):
        self.model = PersonDetSMSV2()

    def __call__(self, img):
        '''
        img: BGR order
        '''
        assert len(img.shape)==3,"Error img size"
        output = self.model(img)
        mask = output[...,-1]==0
        output = output[mask]
        bboxes = output[...,:4]
        #labels = output[...,-1]
        probs = output[...,4]*output[...,5]

        wh = bboxes[...,2:]-bboxes[...,:2]
        wh_mask = wh>3
        size_mask = np.logical_and(wh_mask[...,0],wh_mask[...,1])
        bboxes = bboxes[size_mask]
        probs = probs[size_mask]

        return bboxes,probs
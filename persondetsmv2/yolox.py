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
import wml_utils as wmlu


curdir_path = osp.dirname(__file__)

class YOLOXDetectionS:
    def __init__(self,name="sportsmotv2.torch") -> None:
        onnx_path = osp.join(wmlu.parent_dir_path_of_file(__file__), "models",name)
        self.device = torch.device("cuda:0")
        print(f"Load {onnx_path}")
        self.model = torch.jit.load(onnx_path).to(self.device)

    @staticmethod
    def preproc(img, input_size, swap=(2, 0, 1)):
        if len(img.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        if math.fabs(r - 1) < 0.001:
            resized_img = img
        else:
            resized_img = cv2.resize(
                img,
                (int(img.shape[1] * r), int(img.shape[0] * r)),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    @staticmethod
    def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            if not detections.size(0):
                continue

            nms_out_index = torchvision.ops.nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    nms_thre,
                )
            detections = detections[nms_out_index]

            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))

        return output

    def __call__(self, img):
        '''

        Args:
            img: BGR order

        Returns:
            ans: [N,7] (xmin,ymin,xmax,ymax,obj_probs,cls_probs,cls),B
        '''
        org_shape = img.shape
        input_shape = (800,1440)
        img,r = self.preproc(img,input_shape)
        img = np.expand_dims(img,axis=0)
        img = torch.from_numpy(img).to(self.device)
        with torch.no_grad():
            output = self.model(img).cpu()
        #output = self.postprocess(output,1,0.45,0.45,class_agnostic=False)
        output = self.postprocess(output,1,0.09,0.7,class_agnostic=False)
        output = output[0]
        if output is None:
            return np.zeros([0,7],dtype=np.float32)
        output = output.numpy()
        output[...,:4] = output[...,:4]/r
        output[...,:2] = np.maximum(output[...,:2],0)
        output[...,2] = np.minimum(output[...,2],org_shape[1])
        output[...,3] = np.minimum(output[...,3],org_shape[0])
        bboxes = output[...,:4]
        wh = bboxes[...,2:]-bboxes[...,:2]
        wh_mask = wh>3
        size_mask = np.logical_and(wh_mask[...,0],wh_mask[...,1])
        output = output[size_mask]

        return output
import numpy as np
import os.path as osp
import wml_utils as wmlu
import torch

class DeepReID():
    def __init__(self, onnx_path=None):
        """
        :param onnx_path:
        """
        if onnx_path is None:
            onnx_path = osp.join(wmlu.parent_dir_path_of_file(__file__), "models","deep_reid.torch")
        print(f"--------onnx_path is {onnx_path} ----------------")
        self.device = "cuda"
        print(f"Load {onnx_path}")
        self.model = torch.jit.load(onnx_path).to(self.device)
        # self.onnx_session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
        # self.input_name = self.get_input_name(self.onnx_session)
        # self.output_name = self.get_output_name(self.onnx_session)
        # print("input_name:{}".format(self.input_name))
        # print("output_name:{}".format(self.output_name))
        self.size = (128, 256)

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def __call__(self, image_numpy):
        '''
        # image_numpy = image.transpose(2, 0, 1)
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy:
        # :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_numpy})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: iimage_numpy})
        image_numpy = np.array(image_numpy, dtype=np.uint8).astype(np.float32)
        image_numpy /= 255
        image_numpy -= (0.485, 0.456, 0.406)
        image_numpy /= (0.229, 0.224, 0.225)
        image_numpy = torch.from_numpy(image_numpy.transpose(0, 3, 1, 2)).to(self.device)
        with torch.no_grad():
            output = self.model(image_numpy).cpu().numpy()
        return output
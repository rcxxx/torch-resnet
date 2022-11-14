# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import cv2
import numpy as np

from model import resnet50

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.base = resnet50()


    def forward(self, x):
        # shape [N, C, H, W]
        feature = self.base(x)
        feature = F.avg_pool2d(feature, (feature.shape[2], feature.shape[3]))
        # shape [N, C]
        feature = torch.flatten(feature, 1)
        feature = F.normalize(feature)
        return feature
    
    def preprocess(_image, input_size, scale_im, mean, std, swap=(2, 0, 1)):
        image = cv2.resize(_image, input_size)
        image = image[:, :, ::-1]
        image = image.astype(np.float32)
        if scale_im:
            image /= 255.0
        if mean is not None:
            image -= mean
        if std is not None:
            image /= std
        image = image.transpose(swap)
        image = np.ascontiguousarray(image, dtype=np.float32)
        return image

    def inference(self, mat):
        t0 = time.time()
        img = self.preprocess(mat,
                         input_size=(224, 224),
                         scale_im=True,
                         mean=[0.486, 0.459, 0.408],
                         std=[0.229, 0.224, 0.225])
        img = torch.tensor(img).float().unsqueeze(0)
        feature = self.forward(img)

        return feature
    
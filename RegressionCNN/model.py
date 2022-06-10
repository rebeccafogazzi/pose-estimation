from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import *
import pretrainedmodels

from util import *

# -----------------------------------------------------------------------------

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # self.model =  pretrainedmodels.__dict__['resnet18'](pretrained='imagenet')
        # self.regression_layer = nn.Sequential(nn.Linear(512, out_params))
        self.model =  pretrainedmodels.__dict__['resnet50'](pretrained='imagenet')
        self.regression_layer = nn.Sequential(nn.Linear(2048, out_params))

    def forward(self, x):
        batch_size ,_,_,_ = x.shape #taking out batch_size from input image
        x = self.model.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x,1).reshape(batch_size,-1) # then reshaping the batch_size
        x = self.regression_layer(x)
        x_transl = x[:, -3:]
        x_rot = compute_rotation_matrix_from_ortho6d(x[:, :6].view(batch_size, -1))

        return x_transl, x_rot

    def compute_rotation_matrix_l2_loss(self, gt_rotation_matrix, predict_rotation_matrix):
        loss_function = nn.MSELoss()
        loss = loss_function(predict_rotation_matrix, gt_rotation_matrix)

        return loss

    def compute_rotation_matrix_geodesic_loss(self, gt_rotation_matrix, predict_rotation_matrix):
        theta = compute_geodesic_distance_from_two_matrices(gt_rotation_matrix, predict_rotation_matrix)
        error = theta.mean()

        return error

# -----------------------------------------------------------------------------

# def model_pretrained(pretrained, requires_grad):

#     model = models.resnet18(progress=True, pretrained=pretrained)
#     # freeze hidden layers
#     if requires_grad == False:
#         for param in model.parameters():
#             param.requires_grad = False
#     # train the hidden layers
#     elif requires_grad == True:
#         for param in model.parameters():
#             param.requires_grad = True
#     # make the regression layer learnable
#     model.fc = nn.Linear(512, 6)

#     return model

# -----------------------------------------------------------------------------


# class ResNet18_pretrained(nn.Module):
#     def __init__(self, num_classes):
#         super(ResNet18_pretrained, self).__init__()
#         self.model_resnet = models.resnet18(pretrained=True)
#         num_ftrs = self.model_resnet.fc.in_features
#         self.model_resnet.fc = nn.Identity()
#         self.fc1 = nn.Linear(num_ftrs, num_classes)
#         # self.fc2 = nn.Linear(num_ftrs, num_classes)

#     def forward(self, x):

#         x = self.model_resnet(x)
#         out1 = self.fc1(x)
#         rot_matrix = compute_rotation_matrix_from_ortho6d(out1)
#         return rot_matrix
#         # out2 = self.fc2(x)
#         # return out1, out2
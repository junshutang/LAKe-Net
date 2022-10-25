from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from utils.fps_utils import *
from .pointnetpp.multikp_pointnet2_category import get_model as PointNetPP

class UMKD(nn.Module):  # input (B, N, 3)
    def __init__(self, kp_num1, kp_num2, kp_num3, pca_dim=8, cate_num=55):
        super(UMKD, self).__init__()
        self.input_dim = 2048
        self.PointEncoder = PointNetPP(kp_num1, kp_num2, kp_num3, cate_num)
        self.opredictor1 = nn.ModuleList([nn.Linear(kp_num1, kp_num1) for i in range(cate_num)])
        self.opredictor2 = nn.ModuleList([nn.Linear(kp_num2, kp_num2) for i in range(cate_num)])
        self.opredictor3 = nn.ModuleList([nn.Linear(kp_num3, kp_num3) for i in range(cate_num)])

    def forward(self, x):
        B = x.size()[0]
        point_feat, feat1, feat2, feat3, cls_score = self.PointEncoder(x)  # (B, k, N)
        category_id = torch.max(cls_score, -1).indices
        key_feat1 = torch.Tensor([]).cuda()
        key_feat2 = torch.Tensor([]).cuda()
        key_feat3 = torch.Tensor([]).cuda()
        for i in range(B):
            offset1 = F.relu(self.opredictor1[category_id[i]](feat1[i].unsqueeze(0).permute(0, 2, 1)))
            key_feat1 = torch.cat((key_feat1, feat1[i] + offset1.permute(0, 2, 1)), 0)
            offset2 = F.relu(self.opredictor2[category_id[i]](feat2[i].unsqueeze(0).permute(0, 2, 1)))
            key_feat2 = torch.cat((key_feat2, feat2[i] + offset2.permute(0, 2, 1)), 0)
            offset3 = F.relu(self.opredictor3[category_id[i]](feat3[i].unsqueeze(0).permute(0, 2, 1)))
            key_feat3 = torch.cat((key_feat3, feat3[i] + offset3.permute(0, 2, 1)), 0)
        
        key_feat1 = F.softmax(key_feat1, -1)
        key_feat2 = F.softmax(key_feat2, -1)
        key_feat3 = F.softmax(key_feat3, -1)

        return key_feat1, key_feat2, key_feat3, cls_score


if __name__ == '__main__':
    inputs = torch.randn((8, 9, 2048)).cuda()
    net = UMKD(128, 64, 32).cuda()
    key_feat1, key_feat2, key_feat3, cls_score = net(inputs)
    print(key_feat1.shape, key_feat2.shape, key_feat3.shape, cls_score.shape)


import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import torch
sys.path.append('models/pointnetpp/')
from pointnet_util import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes1, num_classes2, num_classes3, num_cate):
        super(get_model, self).__init__()

        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 9, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.sa5 = PointNetSetAbstraction(None, None, None, 512+512+3, [256, 512, 1024], True)
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        
        self.fc = nn.Linear(1024, 256)
        self.bn = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_cate)

        self.conv1 = nn.Conv1d(128, num_classes1, 1)
        self.bn1 = nn.BatchNorm1d(num_classes1)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(num_classes1, num_classes2, 1)
        self.bn2 = nn.BatchNorm1d(num_classes2)
        self.drop2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv1d(num_classes2, num_classes3, 1)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points) #256
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) # 1024
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points) # 2048

        #cls
        # l5_points, _ = torch.max(l4_points, 2)
        l5_points = l5_points.view(B, 1024)
        cls_x = self.drop(F.relu(self.bn(self.fc(l5_points))))
        cls_x = self.fc2(cls_x)
        cls_x = F.log_softmax(cls_x, -1)

        x1 = F.log_softmax(self.conv1(l0_points), dim=1)
        x2 = F.log_softmax(self.conv2(self.drop1(F.relu(self.bn1(x1)))), dim=1)
        x3 = F.log_softmax(self.conv3(self.drop2(F.relu(self.bn2(x2)))), dim=1)
                
        return l0_points, x1, x2, x3, cls_x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    x, points = model(xyz)
    print(x.shape, points.shape)

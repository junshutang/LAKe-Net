from __future__ import print_function
import os
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import pointnet2_utils as pn2
from utils.model_utils import calc_cd
from utils.snow_utils import MLP_Res, MLP_CONV
from utils.triangle_utils import *
from models.category import UMKD

sys.path.append('models/pointnetpp/')
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(proj_dir, "models"))

def fps(point, num):
    # b, D, N
    pid_fps = pn2.furthest_point_sample(point.transpose(1, 2).contiguous(), num)
    x_fps = pn2.gather_operation(point, pid_fps)

    return x_fps # b, D, Nf

def fps_d(point, num):
    # b, N, D
    pid_fps = pn2.furthest_point_sample(point, num)
    x_fps = pn2.gather_operation(point.transpose(1, 2).contiguous(), pid_fps).transpose(1, 2).contiguous()

    return x_fps # b, Nf, D


class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)

class MLP_Res(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out

class PCN_encoder(nn.Module):
    def __init__(self, output_size=1024):
        super(PCN_encoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(512, 512, 1)
        self.conv4 = nn.Conv1d(512, output_size, 1)

    def forward(self, x):
        batch_size, _, num_points = x.size()
        x1 = F.relu(self.conv1(x)) # b, 128, N
        x2 = self.conv2(x1) # b, 256, N
        global_feature, _ = torch.max(x2, 2) # b, 256
        x3 = torch.cat((x2, global_feature.view(batch_size, -1, 1).repeat(1, 1, num_points).contiguous()), 1) # b, 512, 2048
        x4 = F.relu(self.conv3(x3)) # b, 512, 2048
        x5 = self.conv4(x4) # x.shape=torch.Size([B, 1024, 2048])
        global_feature, _ = torch.max(x5, 2) # b, 1024
        return x5, global_feature.view(batch_size, -1)


class SeedGenerator(nn.Module):
    def __init__(self, dim_feat=1024, num_pc=1024):
        super(SeedGenerator, self).__init__()
        self.ps = nn.ConvTranspose1d(dim_feat, 128, num_pc, bias=True)
        self.mlp_1 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat):
        """
        Args:
            feat: Tensor (b, dim_feat)
        """
        feat = feat.unsqueeze(-1)
        x1 = self.ps(feat)  # (b, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  # (b, 128, 256)
        completion = self.mlp_4(x3)  # (b, 3, 256)
        return completion.transpose(1,2).contiguous()

class MultiKPGenerator(nn.Module):
    def __init__(self, dim_feat=1024, num_kp1=256, num_kp2=128, num_kp3=64):
        super(MultiKPGenerator, self).__init__()
        
        self.num_kp1 = num_kp1
        self.num_kp2 = num_kp2
        self.num_kp3 = num_kp3

        self.ps1 = nn.ConvTranspose1d(dim_feat, 128, num_kp1, bias=True)
        self.ps2 = nn.ConvTranspose1d(dim_feat, 128, num_kp2, bias=True)
        self.ps3 = nn.ConvTranspose1d(dim_feat, 128, num_kp3, bias=True)

        self.mlp_1 = MLP_Res(in_dim=dim_feat * 2 + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )
        self.mlp_3 = MLP_Res(in_dim=dim_feat * 2 + 128, hidden_dim=128, out_dim=128)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )
        self.mlp_5 = MLP_Res(in_dim=dim_feat * 2 + 128, hidden_dim=128, out_dim=128)
        self.mlp_6 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, point_feat, global_feat):
        """
        Args:
            point_feat: Tensor (b, 1024, 2048)
            global_feat: Tensor (b, 1024)
        """
        feat_fps1 = fps(point_feat, self.num_kp1) # (b, 1024, 256)
        feat_fps2 = fps(point_feat, self.num_kp2) # (b, 1024, 128)
        feat_fps3 = fps(point_feat, self.num_kp3) # (b, 1024, 64)

        global_feat = global_feat.unsqueeze(-1) # (b, 1024, 1)
        
        x1 = self.ps1(global_feat)  # (b, 128, 256)
        x2 = self.ps2(global_feat)  # (b, 128, 128)
        x3 = self.ps3(global_feat)  # (b, 128, 64)

        x1 = self.mlp_1(torch.cat([x1, feat_fps1, global_feat.repeat((1, 1, x1.size(2)))], 1)) # (b, 128, 256)
        x2 = self.mlp_3(torch.cat([x2, feat_fps2, global_feat.repeat((1, 1, x2.size(2)))], 1)) # (b, 128, 128)
        x3 = self.mlp_5(torch.cat([x3, feat_fps3, global_feat.repeat((1, 1, x3.size(2)))], 1)) # (b, 128, 64)

        pred_kp1 = self.mlp_2(x1) # (b, 3, 256)
        pred_kp2 = self.mlp_4(x2) # (b, 3, 128)
        pred_kp3 = self.mlp_6(x3) # (b, 3, 64)

        return pred_kp1.transpose(1,2).contiguous(), pred_kp2.transpose(1,2).contiguous(), pred_kp3.transpose(1,2).contiguous()


class Multi_Offset_Predictor(nn.Module):
    def __init__(self, dim_input=1024, dim_output=256, up_factor=4, i=0, radius=1):
        super(Multi_Offset_Predictor, self).__init__()
        self.up_factor = up_factor
        self.i = i
        self.radius = radius

        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2 + dim_input, layer_dims=[256, dim_output])
        self.mlp_res = MLP_Res(in_dim=dim_output, hidden_dim=64, out_dim=dim_output)
        self.mlp_3 = MLP_CONV(in_channel=dim_output, layer_dims=[dim_output//2, 64, 3])

        # self.ps = nn.ConvTranspose1d(128, dim_output, up_factor, up_factor, bias=False)   # point-wise splitting

    def forward(self, pcd_prev, global_feat):
        # pcd_prev: (B, N_prev, 3), gloal_feat: (B, N_feat)
        batchsize, num_point, _ = pcd_prev.shape
        global_feat = global_feat.unsqueeze(2)
        pcd_up = pcd_prev.repeat(1, self.up_factor, 1)

        feat_prev = self.mlp_1(pcd_up.transpose(1,2).contiguous()) # [B, 128, N]
        feat_1 = torch.cat([feat_prev, torch.max(feat_prev, 2, keepdim=True)[0].repeat((1, 1, num_point * self.up_factor)), global_feat.repeat(1, 1, num_point*self.up_factor)], 1)
        feat_up = self.mlp_2(feat_1) # [B, 128, 2048]
        # feat_up = self.ps(feat_2)
        curr_feat = self.mlp_res(feat_up)
        delta = torch.tanh(self.mlp_3(F.relu(curr_feat))) / self.radius**self.i
        fine_pcd = pcd_prev.repeat(1, self.up_factor, 1) + delta.transpose(1,2).contiguous()  # or upsample [B, N * up_factor, 3]
        return fine_pcd, curr_feat.transpose(1,2).contiguous()


class Model(nn.Module):

    def __init__(self, args, num_coarse=1024, size_z=128, up_factors=None, dim_feat=[1024, 512, 256, 128]):
        super(Model, self).__init__()

        self.num_coarse = num_coarse
        self.num_kpcoarse = args.num_kpcoarse
        self.num_points = args.num_points
        self.num_kp1 = args.num_kp1
        self.num_kp2 = args.num_kp2
        self.num_kp3 = args.num_kp3
        self.pretrain_epoch = args.pretrain_epoch
        self.size_z = size_z

        self.kp_detect = UMKD(self.num_kp1, self.num_kp2, self.num_kp3)
        self.encoder = PCN_encoder(dim_feat[0])
        self.partial_encode = PCN_encoder(dim_feat[0])
        self.coarse_decoder = SeedGenerator()
        self.kp_generator = MultiKPGenerator(dim_feat=dim_feat[0], num_kp1=self.num_kp1, num_kp2=self.num_kp2, num_kp3=self.num_kp3)
        self.feat_loss = nn.MSELoss()

        if up_factors is None:
            up_factors = [1]
        else:
            up_factors = [1] + up_factors

        decoders = []
        for i, up_factor in enumerate(up_factors):
            decoders.append(Multi_Offset_Predictor(dim_input=dim_feat[i], dim_output=dim_feat[i+1], up_factor=up_factor, i=i, radius=1))

        self.decoder = nn.ModuleList(decoders)


    def forward(self, x, gt, gt_2048, label, is_training=True, epochidx=None, alpha=None, logging=None):
        batchsize, _, num_input = x.shape
        # gt.size = [B, N, 3] x.size = [B, N, 3]
        
        x_fps = fps_d(x, self.num_kpcoarse)
        gt_fps = fps_d(gt, self.num_kpcoarse)
        
        cat_gt = torch.cat([gt_2048, gt_2048, gt_2048], 2).transpose(1,2).contiguous() # [B, 9, N]
        cat_x = torch.cat([x, x, x], 2).transpose(1,2).contiguous() # [B, 9, N]

        if is_training:
            ######################################
            # pre-train keypoint predict network
            ######################################
            if epochidx is not None and epochidx < self.pretrain_epoch:
                ckp1_w, ckp2_w, ckp3_w, c_cls = self.kp_detect(cat_gt)
                ckp1 = ckp1_w.bmm(gt_2048)
                ckp2 = ckp2_w.bmm(gt_2048)
                ckp3 = ckp3_w.bmm(gt_2048)

                _, c_global_feature = self.encoder(gt.transpose(1,2).contiguous())

                c_A1 = init_graph(gt, ckp1)
                c_A2 = init_graph(gt, ckp2)
                c_A3 = init_graph(gt, ckp3)

                gt_coarse = self.coarse_decoder(c_global_feature)
                c_kp1_coarse1, c_kp1_coarse2 = Batch_Keypoint_graph_interpolation_v2(ckp1, c_A1, self.num_kpcoarse*8, self.num_kpcoarse*8)
                c_kp2_coarse1, c_kp2_coarse2 = Batch_Keypoint_graph_interpolation_v2(ckp2, c_A2, self.num_kpcoarse*8, self.num_kpcoarse*4)
                c_kp3_coarse1, c_kp3_coarse2 = Batch_Keypoint_graph_interpolation_v2(ckp3, c_A3, self.num_kpcoarse*8, self.num_kpcoarse*2)
                c_kp_coarses = [c_kp3_coarse2, c_kp2_coarse2, c_kp1_coarse2]
                pcd = gt_coarse
                global_feat = c_global_feature
                for i, decoder in enumerate(self.decoder):
                    pcd = torch.cat([pcd, c_kp_coarses[i]], dim=1)
                    pcd, point_feat = decoder(pcd, global_feat)
                    global_feat, _ = torch.max(point_feat, 1)
                cout1, cout2 = gt_coarse, pcd
                
                closs1 = calc_cd(cout1, gt)[0] 
                closs2 = calc_cd(cout2, gt)[0]
                loss_kp = calc_cd(ckp1, gt_fps)[0] + calc_cd(ckp2, gt_fps)[0] + calc_cd(ckp3, gt_fps)[0]  + \
                          calc_cd(c_kp1_coarse1, gt)[0] + calc_cd(c_kp2_coarse1, gt)[0] + calc_cd(c_kp3_coarse1, gt)[0]
            
                # Classification Loss
                loss_cls = F.nll_loss(c_cls, label)
                train_loss = closs1.mean() + closs2.mean() + loss_kp.mean() + loss_cls.mean() * 10

                return cout1, cout2, closs1, closs2, loss_kp, loss_cls, train_loss
            
            ######################################
            # train completion network 
            ######################################
            elif epochidx is not None and epochidx >= self.pretrain_epoch:

                ckp1_w, ckp2_w, ckp3_w, _ = self.kp_detect(cat_gt)
                ckp1 = ckp1_w.bmm(gt_2048)
                ckp2 = ckp2_w.bmm(gt_2048)
                ckp3 = ckp3_w.bmm(gt_2048)

                _, c_global_feature = self.encoder(gt.transpose(1,2).contiguous())
                p_point_feature, p_global_feature = self.partial_encode(x.transpose(1,2).contiguous())
                pred_coarse = self.coarse_decoder(p_global_feature) 
                new_pred_coarse = fps_d(torch.cat([pred_coarse, x], 1), self.num_coarse) 
                pred_kp1, pred_kp2, pred_kp3 = self.kp_generator(p_point_feature, p_global_feature)
                
                p_A1 = init_graph(new_pred_coarse, pred_kp1)
                p_A2 = init_graph(new_pred_coarse, pred_kp2)
                p_A3 = init_graph(new_pred_coarse, pred_kp3)
                p_kp1_coarse1, p_kp1_coarse2 = Batch_Keypoint_graph_interpolation_v2(pred_kp1, p_A1, self.num_kpcoarse*8, self.num_kpcoarse*8)
                p_kp2_coarse1, p_kp2_coarse2 = Batch_Keypoint_graph_interpolation_v2(pred_kp2, p_A2, self.num_kpcoarse*8, self.num_kpcoarse*4)
                p_kp3_coarse1, p_kp3_coarse2 = Batch_Keypoint_graph_interpolation_v2(pred_kp3, p_A3, self.num_kpcoarse*8, self.num_kpcoarse*2)
                
                p_kp_coarses = [p_kp3_coarse2, p_kp2_coarse2, p_kp1_coarse2]
                pcd = new_pred_coarse
                global_feat = p_global_feature
                for i, decoder in enumerate(self.decoder):
                    pcd = torch.cat([pcd, p_kp_coarses[i]], dim=1)
                    pcd, point_feat = decoder(pcd, global_feat)
                    global_feat, _ = torch.max(point_feat, 1)
                pout1, pout2 = new_pred_coarse, pcd

                # ############# Compute Loss ##############
                # CD loss
                ploss1, _ = calc_cd(pout1, gt) #[B, N, 3]
                ploss2, _ = calc_cd(pout2, gt)

                # KP loss
                distance1 = (ckp1 - pred_kp1).norm(dim=2).sum(dim=1) / self.num_kp1
                distance2 = (ckp2 - pred_kp2).norm(dim=2).sum(dim=1) / self.num_kp2
                distance3 = (ckp3 - pred_kp3).norm(dim=2).sum(dim=1) / self.num_kp3
                kp_com_loss = distance1 +  distance2 + distance3
                kp_loss = calc_cd(pred_kp1, gt_fps)[0] + calc_cd(pred_kp2, gt_fps)[0] + calc_cd(pred_kp3, gt_fps)[0]
                kp_coarse_loss = calc_cd(p_kp1_coarse1, gt)[0] + calc_cd(p_kp2_coarse1, gt)[0] + calc_cd(p_kp3_coarse1, gt)[0]
                loss_kpcom = kp_com_loss + kp_loss + kp_coarse_loss
                # loss_kpcom = kp_loss 
                # Feature match loss:
                # loss_feat = 1 - torch.cosine_similarity(p_global_feature, c_global_feature, dim=1)
                loss_feat = self.feat_loss(p_global_feature, c_global_feature)
                total_train_loss = ploss1.mean() + ploss2.mean() + loss_kpcom.mean() * 10 + loss_feat * 1000

                return pout1, pout2, ploss1, ploss2, kp_com_loss, kp_loss, kp_coarse_loss, loss_feat, total_train_loss
        
        else:
            ######################################
            # pre-train keypoint predict network
            ######################################
            if epochidx is not None and epochidx < self.pretrain_epoch:

                ckp1_w, ckp2_w, ckp3_w, c_cls = self.kp_detect(cat_gt)
                ckp1 = ckp1_w.bmm(gt_2048)
                ckp2 = ckp2_w.bmm(gt_2048)
                ckp3 = ckp3_w.bmm(gt_2048)

                _, c_global_feature = self.encoder(gt.transpose(1,2).contiguous())

                c_A1 = init_graph(gt, ckp1)
                c_A2 = init_graph(gt, ckp2)
                c_A3 = init_graph(gt, ckp3)

                gt_coarse = self.coarse_decoder(c_global_feature)
                c_kp1_coarse1, c_kp1_coarse2 = Batch_Keypoint_graph_interpolation_v2(ckp1, c_A1, self.num_kpcoarse*8, self.num_kpcoarse*8)
                c_kp2_coarse1, c_kp2_coarse2 = Batch_Keypoint_graph_interpolation_v2(ckp2, c_A2, self.num_kpcoarse*8, self.num_kpcoarse*4)
                c_kp3_coarse1, c_kp3_coarse2 = Batch_Keypoint_graph_interpolation_v2(ckp3, c_A3, self.num_kpcoarse*8, self.num_kpcoarse*2)
                c_kp_coarses = [c_kp3_coarse2, c_kp2_coarse2, c_kp1_coarse2]
                pcd = gt_coarse
                global_feat = c_global_feature
                for i, decoder in enumerate(self.decoder):
                    pcd = torch.cat([pcd, c_kp_coarses[i]], dim=1)
                    pcd, point_feat = decoder(pcd, global_feat)
                    global_feat, _ = torch.max(point_feat, 1)
                cout1, cout2 = gt_coarse, pcd

                cd_p, cd_t, f1 = calc_cd(cout2, gt, calc_f1=True)
                return {'kp1': ckp1, 'kp2': ckp2, 'kp3': ckp3, 'kpc1': c_kp1_coarse1, 'kpc2': c_kp2_coarse1, 'kpc3': c_kp3_coarse1, 'out1': cout1, 'out2': cout2, 'cd_p': cd_p, 'cd_t': cd_t, 'f1': f1}

            ######################################
            # train completion network 
            ######################################
            elif epochidx is not None and epochidx >= self.pretrain_epoch:

                p_point_feature, p_global_feature = self.partial_encode(x.transpose(1,2).contiguous())
                pred_coarse = self.coarse_decoder(p_global_feature) 
                new_pred_coarse = fps_d(torch.cat([pred_coarse, x], 1), self.num_coarse) 
                pred_kp1, pred_kp2, pred_kp3 = self.kp_generator(p_point_feature, p_global_feature)
                
                p_A1 = init_graph(new_pred_coarse, pred_kp1)
                p_A2 = init_graph(new_pred_coarse, pred_kp2)
                p_A3 = init_graph(new_pred_coarse, pred_kp3)
                p_kp1_coarse1, p_kp1_coarse2 = Batch_Keypoint_graph_interpolation_v2(pred_kp1, p_A1, self.num_kpcoarse*8, self.num_kpcoarse*8)
                p_kp2_coarse1, p_kp2_coarse2 = Batch_Keypoint_graph_interpolation_v2(pred_kp2, p_A2, self.num_kpcoarse*8, self.num_kpcoarse*4)
                p_kp3_coarse1, p_kp3_coarse2 = Batch_Keypoint_graph_interpolation_v2(pred_kp3, p_A3, self.num_kpcoarse*8, self.num_kpcoarse*2)
                
                p_kp_coarses = [p_kp3_coarse2, p_kp2_coarse2, p_kp1_coarse2]
                pcd = new_pred_coarse
                global_feat = p_global_feature
                for i, decoder in enumerate(self.decoder):
                    pcd = torch.cat([pcd, p_kp_coarses[i]], dim=1)
                    pcd, point_feat = decoder(pcd, global_feat)
                    global_feat, _ = torch.max(point_feat, 1)
                pout1, pout2 = new_pred_coarse, pcd

                # emd = calc_emd(pout2, gt, eps=0.004, iterations=3000)
                cd_p, cd_t, f1 = calc_cd(pout2, gt, calc_f1=True)
                return {'kp1': pred_kp1, 'kp2': pred_kp2, 'kp3': pred_kp3, 'kpc1': p_kp1_coarse1, 'kpc2': p_kp2_coarse1, 'kpc3': p_kp3_coarse1, 'out1': pout1, 'out2': pout2, 'cd_p': cd_p, 'cd_t': cd_t, 'f1': f1}

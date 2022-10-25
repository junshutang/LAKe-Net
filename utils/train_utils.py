import random   
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

class AverageValueMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def save_model(path, net, net_d=None):
    if net_d is not None:
        torch.save({'net_state_dict': net.module.state_dict(),
                    'D_state_dict': net_d.module.state_dict()}, path)
    else:
        torch.save({'net_state_dict': net.state_dict()}, path)


def generator_step(net_d, out2, net_loss, optimizer):
    set_requires_grad(net_d, False)
    d_fake = net_d(out2[:, 0:2048, :])
    errG_loss_batch = torch.mean((d_fake - 1) ** 2)
    total_gen_loss_batch = errG_loss_batch + net_loss * 200
    total_gen_loss_batch.backward(torch.ones(torch.cuda.device_count()).cuda(), retain_graph=True, )
    optimizer.step()
    return d_fake


def discriminator_step(net_d, gt, d_fake, optimizer_d):
    set_requires_grad(net_d, True)
    d_real = net_d(gt[:, 0:2048, :])
    d_loss_fake = torch.mean(d_fake ** 2)
    d_loss_real = torch.mean((d_real - 1) ** 2)
    errD_loss_batch = 0.5 * (d_loss_real + d_loss_fake)
    total_dis_loss_batch = errD_loss_batch
    total_dis_loss_batch.backward(torch.ones(torch.cuda.device_count()).cuda())
    optimizer_d.step()

def seprate_point_cloud(xyz, num_points, crop, fixed_points = None, padding_zeros = False):
    '''
     seprate point cloud: usage : using to generate the incomplete point cloud with a setted number.
    '''
    _,n,c = xyz.shape

    assert n == num_points
    assert c == 3
    if crop == num_points:
        return xyz, None
        
    INPUT = []
    CROP = []
    for points in xyz:
        if isinstance(crop,list):
            num_crop = random.randint(crop[0],crop[1])
        else:
            num_crop = crop

        points = points.unsqueeze(0)

        if fixed_points is None:       
            center = F.normalize(torch.randn(1,1,3),p=2,dim=-1).cuda()
        else:
            if isinstance(fixed_points,list):
                fixed_point = random.sample(fixed_points,1)[0]
            else:
                fixed_point = fixed_points
            center = fixed_point.reshape(1,1,3).cuda()

        distance_matrix = torch.norm(center.unsqueeze(2) - points.unsqueeze(1), p =2 ,dim = -1)  # 1 1 2048

        idx = torch.argsort(distance_matrix,dim=-1, descending=False)[0,0] # 2048

        if padding_zeros:
            input_data = points.clone()
            input_data[0, idx[:num_crop]] =  input_data[0,idx[:num_crop]] * 0

        else:
            input_data = points.clone()[0, idx[num_crop:]].unsqueeze(0) # 1 N 3

        crop_data =  points.clone()[0, idx[:num_crop]].unsqueeze(0)

        if isinstance(crop,list):
            INPUT.append(fps(input_data,2048))
            CROP.append(fps(crop_data,2048))
        else:
            INPUT.append(input_data)
            CROP.append(crop_data)

    input_data = torch.cat(INPUT,dim=0)# B N 3
    crop_data = torch.cat(CROP,dim=0)# B M 3

    return input_data.contiguous(), crop_data.contiguous()





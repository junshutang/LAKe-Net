import numpy as np
import torch
import copy
import utils.DistFunc as DF
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import *


def Batch_calculate_area(closest_keypoints):  # (B, n, 3, 3)
    closest_keypoints = closest_keypoints.double()

    side_ab = torch.sqrt((closest_keypoints[..., 0, 0] - closest_keypoints[..., 1, 0]) ** 2 + \
                         (closest_keypoints[..., 0, 1] - closest_keypoints[..., 1, 1]) ** 2 + \
                         (closest_keypoints[..., 0, 2] - closest_keypoints[..., 1, 2]) ** 2)

    side_ac = torch.sqrt((closest_keypoints[..., 0, 0] - closest_keypoints[..., 2, 0]) ** 2 + \
                         (closest_keypoints[..., 0, 1] - closest_keypoints[..., 2, 1]) ** 2 + \
                         (closest_keypoints[..., 0, 2] - closest_keypoints[..., 2, 2]) ** 2)

    side_bc = torch.sqrt((closest_keypoints[..., 1, 0] - closest_keypoints[..., 2, 0]) ** 2 + \
                         (closest_keypoints[..., 1, 1] - closest_keypoints[..., 2, 1]) ** 2 + \
                         (closest_keypoints[..., 1, 2] - closest_keypoints[..., 2, 2]) ** 2)

    s = (side_ab + side_bc + side_ac) / 2
    # print(s.dtype)
    # area = torch.sqrt(s.float() * (s - side_ab).float()).double() * \
    #        torch.sqrt((s - side_bc).float() * (s - side_ac).float()).double()

    area = (s * (s - side_ab) * (s - side_bc) * (s - side_ac) + 1e-14) ** 0.5
    # print(area.dtype)

    return area.float(), side_ab.float(), side_bc.float(), side_ac.float()  # (B, n)


# batch_index_select
# t (B, n, 3), inds (B, n ,k), return (B, n ,k ,3)
def Batch_index_select(t, dim, inds):
    t = t.unsqueeze(1).expand(t.size(0), t.size(1), t.size(1), t.size(2))  # expand t to (B,n,n,3)
    dummy = inds.unsqueeze(-1).expand(inds.size(0), inds.size(1), inds.size(2), t.size(3))
    out = t.gather(dim, dummy)
    return out


# list_index_select
# t (B, n, 3), ind: list of (?, 3), return: list of (?, 3, 3)
def list_index_select(t, inds):
    bn = t.size()[0]
    triangle = []
    for i in range(bn):
        t_i = t[i]
        triangle_i = t_i[inds[i]]
        triangle.append(triangle_i)
    return triangle


# compute Batch Euclidean Distances of each pair of rows in two tensor a & b
# a (b, m, k); b (b, n, k); output (b, m, n)
def Batch_EuclideanDistances(a, b):
    sq_a = a ** 2
    sum_sq_a = torch.sum(sq_a, dim=2).unsqueeze(2)  # m->[m, 1]
    # print(sum_sq_a.shape)
    sq_b = b ** 2
    sum_sq_b = torch.sum(sq_b, dim=2).unsqueeze(1)  # n->[1, n]
    # print(sum_sq_b.shape)
    bt = b.permute(0, 2, 1)
    return (sum_sq_a + sum_sq_b - 2 * a.bmm(bt) + 1e-14) ** 0.5  # [m, 1] + [1, n] -> [m, n]


# generate triangular interpolation from keypoints(using Batch_EuclideanDistances)
def Batch_Keypoint_interpolation(batch_keypoint, max_n):  # batch_keypoint: (B, n, 3)
    batchresults = torch.Tensor([]).cuda()
    Dis_mat = Batch_EuclideanDistances(batch_keypoint.double(), batch_keypoint.double())  # (B, n, n)
    vertices = torch.topk(Dis_mat, 3, dim=-1, largest=False).indices  # closest vertices:(B, n, 3)
    # print(vertices.shape)
    closest_keypoints = Batch_index_select(batch_keypoint, 2, vertices)  # (B, n, 3, 3)
    return Batch_interp_triangle(closest_keypoints, 0.01, max_n)


def init_graph(shape_xyz, skel_xyz, valid_k=8):
    bn, pn = skel_xyz.size()[0], skel_xyz.size()[1]

    knn_skel = DF.knn_with_batch(skel_xyz, skel_xyz, pn, is_max=False)
    knn_sp2sk = DF.knn_with_batch(shape_xyz, skel_xyz, 3, is_max=False)

    A = torch.zeros((bn, pn, pn)).float().cuda()

    # initialize A with recovery prior: Mark A[i,j]=1 if (i,j) are two skeletal points closest to a surface point
    A[torch.arange(bn)[:, None], knn_sp2sk[:, :, 0], knn_sp2sk[:, :, 1]] = 1
    A[torch.arange(bn)[:, None], knn_sp2sk[:, :, 1], knn_sp2sk[:, :, 0]] = 1

    # initialize A with topology prior
    A[torch.arange(bn)[:, None, None], torch.arange(pn)[None, :, None], knn_skel[:, :, 1:3]] = 1
    A[torch.arange(bn)[:, None, None], knn_skel[:, :, 1:3], torch.arange(pn)[None, :, None]] = 1

    A = torch.triu(A, diagonal=1) + torch.triu(A.transpose(1,2), diagonal=1).transpose(1,2)

    return A

def init_self_graph(skel_xyz, valid_k=8):

    bn, pn = skel_xyz.size()[0], skel_xyz.size()[1]
    knn_skel = DF.knn_with_batch(skel_xyz, skel_xyz, pn, is_max=False)

    A = torch.zeros((bn, pn, pn)).float().cuda()
    
    # initialize A with topology prior
    A[torch.arange(bn)[:, None, None], torch.arange(pn)[None, :, None], knn_skel[:, :, 1:2]] = 1
    A[torch.arange(bn)[:, None, None], knn_skel[:, :, 1:2], torch.arange(pn)[None, :, None]] = 1

    # valid mask: known existing links + knn links
    valid_mask = copy.deepcopy(A)
    valid_mask[torch.arange(bn)[:, None, None], torch.arange(pn)[None, :, None], knn_skel[:, :, 1:valid_k]] = 1
    valid_mask[torch.arange(bn)[:, None, None], knn_skel[:, :, 1:valid_k], torch.arange(pn)[None, :, None]] = 1

    # known mask: known existing links + known absent links, used as the mask to compute binary loss
    known_mask = copy.deepcopy(A)
    known_indice = list(range(valid_k, pn))
    known_mask[torch.arange(bn)[:, None, None], torch.arange(pn)[None, :, None], knn_skel[:, :, known_indice]] = 1
    known_mask[torch.arange(bn)[:, None, None], knn_skel[:, :, known_indice], torch.arange(pn)[None, :, None]] = 1

    A = torch.triu(A, diagonal=1) + torch.triu(A.transpose(1,2), diagonal=1).transpose(1,2)
    valid_mask = torch.triu(valid_mask, diagonal=1) + torch.triu(valid_mask.transpose(1,2), diagonal=1).transpose(1,2)
    known_mask = torch.triu(known_mask, diagonal=1) + torch.triu(known_mask.transpose(1,2), diagonal=1).transpose(1,2)

    return A, valid_mask, known_mask


def batch_triangle_num(A):  # calculate the max triangle number in batch, so other samples need to pad to this size
    A_2 = torch.bmm(A, A)
    A_3 = torch.bmm(A_2, A)
    count = torch.diagonal(A_3, dim1=-2, dim2=-1).sum(1) / 6  # (B), triangle num of each sample
    count = count.long()
    max_num = torch.max(count).item()
    # print("count:", count)
    # print("max_num:", max_num)
    return count, max_num


def batch_triangle_line_num(A):  # calculate the max triangle number in batch, so other samples need to pad to this size
    # print("A: ", A)
    A_2 = torch.bmm(A, A)
    A_3 = torch.bmm(A_2, A)
    tri_count = torch.diagonal(A_3, dim1=-2, dim2=-1).sum(1) / 6  # (B), triangle num of each sample
    # print("tri_count:", tri_count)
    triu_A = torch.triu(A)
    line_count = triu_A.sum((2, 1))
    # print("line_count:", line_count)
    count = (tri_count + line_count).long()
    max_num = torch.max(count).item()
    # print("count:", count)
    # print("max_num:", max_num)
    return count, max_num

def batch_line_num(A):  # calculate the max triangle number in batch, so other samples need to pad to this size
    # print("A: ", A)
    triu_A = torch.triu(A)
    line_count = triu_A.sum((2, 1))
    # print("line_count:", line_count)
    count = line_count.long()
    max_num = torch.max(count).item()
    # print("count:", count)
    # print("max_num:", max_num)
    return count, max_num


def generate_triangle_vertices_v2(keypoint, A):
    # print(A)
    Bn = A.size()[0]
    kp_num = A.size()[1]
    triu_A = torch.triu(A)
    count, max_num = batch_triangle_line_num(A)
    vertices = torch.LongTensor([]).cuda()
    # print(count, max_num)
    '''generate triangle vertices'''
    edge_indices = triu_A.nonzero(as_tuple=False)
    u = edge_indices[:, [0, 1]]  # first index
    v = edge_indices[:, [0, 2]]  # second index
    e = u * torch.Tensor([[1, kp_num]]).cuda() + v * torch.Tensor([[0, 1]]).cuda()  # edge index (u * kp_num + v)
    e = e[:, 1].long()
    # u_e = torch.cat((u, e[:, 1].unsqueeze(1)), -1)
    # u_v = torch.cat((v, e[:, 1].unsqueeze(1)), -1)
    batch_index = u[:, 0].long()
    u_index = u[:, 1].long()
    v_index = v[:, 1].long()
    # print("batch_index: ", batch_index)
    edge_matrix = torch.zeros(Bn, kp_num, kp_num * kp_num).cuda()
    edge_matrix = edge_matrix.index_put([batch_index, u_index, e], torch.tensor(1.).cuda())
    edge_matrix = edge_matrix.index_put([batch_index, v_index, e], torch.tensor(1.).cuda())
    # print("edge_matrix: ", edge_matrix)
    final_matrix = torch.bmm(A, edge_matrix)
    tri_position = (final_matrix == 2).nonzero(as_tuple=False)  # return indices where items equal to 2.
    # print("tri_position: ", tri_position)
    u_origin = torch.div(tri_position[:, 2].float(), kp_num).floor().long()
    v_origin = torch.remainder(tri_position[:, 2], kp_num)
    triangle_vertices = torch.cat((tri_position[:, :2], u_origin.unsqueeze(1), v_origin.unsqueeze(1)), 1)

    '''generate skeleton vertices, and expand (a,b) to (a,b,b) to fit the format of triangle vertices'''
    line_vertices = torch.cat((edge_indices, edge_indices[:, 2].unsqueeze(1)), -1)
    line_batch_info = line_vertices[:, 0]  # items represent batch num
    triangle_batch_info = triangle_vertices[:, 0]  # items represent batch num
    # print("info:", triangle_batch_info)

    for i in range(Bn):
        triangle_batch_i = (triangle_batch_info == i).nonzero(as_tuple=False).squeeze(-1)
        # print(triangle_batch_i)
        triangle_vertices_i = triangle_vertices[triangle_batch_i]  # triangle vertices of batch i
        # print("triangle_vertices: ", triangle_vertices_i.shape)
        line_batch_i = (line_batch_info == i).nonzero(as_tuple=False).squeeze(-1)
        line_vertices_i = line_vertices[line_batch_i]  # line vertices of batch i

        all_vertices_i = torch.cat((triangle_vertices_i[:, 1:], line_vertices_i[:, 1:]), 0)
        all_vertices_i, _ = torch.sort(all_vertices_i, dim=-1)
        all_vertices_i = all_vertices_i.unique(dim=0)
        # print("all_vertices_i:", all_vertices_i.shape)
        all_vertices_i = torch.cat(
            (all_vertices_i, torch.LongTensor([[0, 0, 0]]).cuda().expand(max_num - count[i].item(), 3)), 0)
        # print("all_vertices_i:", all_vertices_i.shape)
        vertices = torch.cat((vertices, all_vertices_i.unsqueeze(0)), 0)
    return vertices.long()


def generate_triangle_vertices_v3(keypoint, A):
    # print(A)
    Bn = A.size()[0]
    kp_num = A.size()[1]
    triu_A = torch.triu(A)
    count, max_num = batch_line_num(A)
    vertices = torch.LongTensor([]).cuda()
    # print(count, max_num)
    '''generate triangle vertices'''
    edge_indices = triu_A.nonzero(as_tuple=False)
    '''generate skeleton vertices, and expand (a,b) to (a,b,b) to fit the format of triangle vertices'''
    line_vertices = torch.cat((edge_indices, edge_indices[:, 2].unsqueeze(1)), -1)
    line_batch_info = line_vertices[:, 0]  # items represent batch num

    for i in range(Bn):

        line_batch_i = (line_batch_info == i).nonzero(as_tuple=False).squeeze(-1)
        line_vertices_i = line_vertices[line_batch_i]  # line vertices of batch i

        all_vertices_i = line_vertices_i[:, 1:]
        all_vertices_i, _ = torch.sort(all_vertices_i, dim=-1)
        all_vertices_i = all_vertices_i.unique(dim=0)

        all_vertices_i = torch.cat(
            (all_vertices_i, torch.LongTensor([[0, 0, 0]]).cuda().expand(max_num - count[i].item(), 3)), 0)
        # print("all_vertices_i:", all_vertices_i.shape)
        vertices = torch.cat((vertices, all_vertices_i.unsqueeze(0)), 0)
    return vertices.long()



def random_sample_triangle_skeleton(triangle, keypoint, interval=0.001, max_n1=2048, max_n2=512):  # (B, n, 3, 3)
    ver_a = triangle[..., 0, :].unsqueeze(2).expand(-1, -1, 100, -1)
    edge_ab = (triangle[..., 1, :] - triangle[..., 0, :]).unsqueeze(2).expand(-1, -1, 100, -1)
    edge_ac = (triangle[..., 2, :] - triangle[..., 0, :]).unsqueeze(2).expand(-1, -1, 100, -1)

    area, side_ab, side_bc, side_ac = Batch_calculate_area(triangle)
    c_square = (side_ab + side_bc + side_ac) ** 2
    c_square = (c_square/ interval).long()
    count = torch.clamp(c_square, 0, 100)  # (B, n)
    count = count.unsqueeze(-1)  # (B, n, 1)

    base_index = torch.arange(100).expand(count.size()[0], count.size()[1], 100).cuda()
    binary_mask = base_index < count  # (B, n, 100)

    random_x = torch.rand(count.size()[0], count.size()[1], 100).cuda()
    random_y = torch.rand(count.size()[0], count.size()[1], 100).cuda()
    x = torch.where(random_x + random_y > 1.0, 1.0 - random_x, random_x)
    y = torch.where(random_x + random_y > 1.0, 1.0 - random_y, random_y)
    final_x = torch.where(binary_mask, x, torch.Tensor([0.0]).cuda()).unsqueeze(-1).expand(-1, -1, -1,
                                                                                           3)  # (B, n, 100, 3)
    final_y = torch.where(binary_mask, y, torch.Tensor([0.0]).cuda()).unsqueeze(-1).expand(-1, -1, -1,
                                                                                           3)  # (B, n, 100, 3)
    final_a = torch.where(binary_mask, torch.Tensor([1.0]).cuda(), torch.Tensor([0.0]).cuda()).unsqueeze(-1).expand(-1,
                                                                                                                    -1,
                                                                                                                    -1,
                                                                                                                    3)  # (B, n, 100, 3)
    coarse = ver_a * final_a + final_x * edge_ab + final_y * edge_ac
    coarse = coarse.reshape(coarse.size()[0], -1, 3)
    # print("coarse shape:", coarse.shape)
    # print("coarse:", coarse)
    final_coarse1 = torch.Tensor([]).cuda()
    final_coarse2 = torch.Tensor([]).cuda()
    for i in range(coarse.size()[0]):
        non_empty_mask = coarse[i].abs().sum(dim=-1).bool()  # filter out
        indices = non_empty_mask.nonzero(as_tuple=False).squeeze(-1)
        coarse_i = torch.cat((coarse[i][indices], keypoint[i]), 0)
        coarse_i = coarse_i.unsqueeze(0)
        # print("coarse_i:", coarse_i.shape)
        # resample
        idx = np.random.permutation(coarse_i.shape[1])
        if idx.shape[0] < max_n1:
            idx = np.concatenate([idx, np.random.randint(coarse_i.shape[1], size=max_n1 - coarse_i.shape[1])])
        coarse_i_1 = coarse_i[:, idx[:max_n1]]
        if idx.shape[0] < max_n2:
            idx = np.concatenate([idx, np.random.randint(coarse_i.shape[1], size=max_n2 - coarse_i.shape[1])])
        coarse_i_2 = coarse_i[:, idx[:max_n2]]
        final_coarse1 = torch.cat((final_coarse1, coarse_i_1), 0)
        final_coarse2 = torch.cat((final_coarse2, coarse_i_2), 0)
    return final_coarse1, final_coarse2


def random_sample_triangle_skeleton_v2(triangle, keypoint, interval=0.001):  # (B, n, 3, 3)
    ver_a = triangle[..., 0, :].unsqueeze(2).expand(-1, -1, 100, -1)
    edge_ab = (triangle[..., 1, :] - triangle[..., 0, :]).unsqueeze(2).expand(-1, -1, 100, -1)
    edge_ac = (triangle[..., 2, :] - triangle[..., 0, :]).unsqueeze(2).expand(-1, -1, 100, -1)

    area, side_ab, side_bc, side_ac = Batch_calculate_area(triangle)
    c_square = (side_ab + side_bc + side_ac) ** 2
    c_square = (c_square/ interval).long()
    count = torch.clamp(c_square, 0, 100)  # (B, n)
    count = count.unsqueeze(-1)  # (B, n, 1)

    base_index = torch.arange(100).expand(count.size()[0], count.size()[1], 100).cuda()
    binary_mask = base_index < count  # (B, n, 100)

    random_x = torch.rand(count.size()[0], count.size()[1], 100).cuda()
    random_y = torch.rand(count.size()[0], count.size()[1], 100).cuda()
    x = torch.where(random_x + random_y > 1.0, 1.0 - random_x, random_x)
    y = torch.where(random_x + random_y > 1.0, 1.0 - random_y, random_y)
    final_x = torch.where(binary_mask, x, torch.Tensor([0.0]).cuda()).unsqueeze(-1).expand(-1, -1, -1,
                                                                                           3)  # (B, n, 100, 3)
    final_y = torch.where(binary_mask, y, torch.Tensor([0.0]).cuda()).unsqueeze(-1).expand(-1, -1, -1,
                                                                                           3)  # (B, n, 100, 3)
    final_a = torch.where(binary_mask, torch.Tensor([1.0]).cuda(), torch.Tensor([0.0]).cuda()).unsqueeze(-1).expand(-1,
                                                                                                                    -1,
                                                                                                                    -1,
                                                                                                                    3)  # (B, n, 100, 3)
    coarse = ver_a * final_a + final_x * edge_ab + final_y * edge_ac
    coarse = coarse.reshape(coarse.size()[0], -1, 3)
    return coarse

def random_sample_triangle_skeleton_v3(triangle, keypoint, interval=0.001, max_n1=2048, max_n2=512, max_n3=256):  # (B, n, 3, 3)
    ver_a = triangle[..., 0, :].unsqueeze(2).expand(-1, -1, 100, -1)
    edge_ab = (triangle[..., 1, :] - triangle[..., 0, :]).unsqueeze(2).expand(-1, -1, 100, -1)
    edge_ac = (triangle[..., 2, :] - triangle[..., 0, :]).unsqueeze(2).expand(-1, -1, 100, -1)

    area, side_ab, side_bc, side_ac = Batch_calculate_area(triangle)
    c_square = (side_ab + side_bc + side_ac) ** 2
    c_square = (c_square/ interval).long()
    count = torch.clamp(c_square, 0, 100)  # (B, n)
    count = count.unsqueeze(-1)  # (B, n, 1)

    base_index = torch.arange(100).expand(count.size()[0], count.size()[1], 100).cuda()
    binary_mask = base_index < count  # (B, n, 100)

    random_x = torch.rand(count.size()[0], count.size()[1], 100).cuda()
    random_y = torch.rand(count.size()[0], count.size()[1], 100).cuda()
    x = torch.where(random_x + random_y > 1.0, 1.0 - random_x, random_x)
    y = torch.where(random_x + random_y > 1.0, 1.0 - random_y, random_y)
    final_x = torch.where(binary_mask, x, torch.Tensor([0.0]).cuda()).unsqueeze(-1).expand(-1, -1, -1,
                                                                                           3)  # (B, n, 100, 3)
    final_y = torch.where(binary_mask, y, torch.Tensor([0.0]).cuda()).unsqueeze(-1).expand(-1, -1, -1,
                                                                                           3)  # (B, n, 100, 3)
    final_a = torch.where(binary_mask, torch.Tensor([1.0]).cuda(), torch.Tensor([0.0]).cuda()).unsqueeze(-1).expand(-1,
                                                                                                                    -1,
                                                                                                                    -1,
                                                                                                                    3)  # (B, n, 100, 3)
    coarse = ver_a * final_a + final_x * edge_ab + final_y * edge_ac
    coarse = coarse.reshape(coarse.size()[0], -1, 3)
    # print("coarse shape:", coarse.shape)
    # print("coarse:", coarse)
    final_coarse1 = torch.Tensor([]).cuda()
    final_coarse2 = torch.Tensor([]).cuda()
    final_coarse3 = torch.Tensor([]).cuda()
    for i in range(coarse.size()[0]):
        non_empty_mask = coarse[i].abs().sum(dim=-1).bool()  # filter out
        indices = non_empty_mask.nonzero(as_tuple=False).squeeze(-1)
        coarse_i = torch.cat((coarse[i][indices], keypoint[i]), 0)
        coarse_i = coarse_i.unsqueeze(0)
        # print("coarse_i:", coarse_i.shape)
        # resample
        idx = np.random.permutation(coarse_i.shape[1])
        if idx.shape[0] < max_n1:
            idx = np.concatenate([idx, np.random.randint(coarse_i.shape[1], size=max_n1 - coarse_i.shape[1])])
        coarse_i_1 = coarse_i[:, idx[:max_n1]]
        if idx.shape[0] < max_n2:
            idx = np.concatenate([idx, np.random.randint(coarse_i.shape[1], size=max_n2 - coarse_i.shape[1])])
        coarse_i_2 = coarse_i[:, idx[:max_n2]]
        if idx.shape[0] < max_n3:
            idx = np.concatenate([idx, np.random.randint(coarse_i.shape[1], size=max_n3 - coarse_i.shape[1])])
        coarse_i_3 = coarse_i[:, idx[:max_n3]]
        final_coarse1 = torch.cat((final_coarse1, coarse_i_1), 0)
        final_coarse2 = torch.cat((final_coarse2, coarse_i_2), 0)
        final_coarse3 = torch.cat((final_coarse3, coarse_i_3), 0)
    return final_coarse1, final_coarse2, final_coarse3


def random_sample_skeleton(triangle, keypoint, interval=0.001, max_n1=2048, max_n2=512):  # (B, n, 3, 3)
    ver_a = triangle[..., 0, :].unsqueeze(2).expand(-1, -1, 100, -1)
    edge_ab = (triangle[..., 1, :] - triangle[..., 0, :]).unsqueeze(2).expand(-1, -1, 100, -1)
    edge_ac = (triangle[..., 2, :] - triangle[..., 0, :]).unsqueeze(2).expand(-1, -1, 100, -1)

    area, side_ab, side_bc, side_ac = Batch_calculate_area(triangle)
    c_square = (side_ab + side_bc + side_ac) ** 2
    c_square = (c_square/ interval).long()
    count = torch.clamp(c_square, 0, 100)  # (B, n)
    count = count.unsqueeze(-1)  # (B, n, 1)

    base_index = torch.arange(100).expand(count.size()[0], count.size()[1], 100).cuda()
    binary_mask = base_index < count  # (B, n, 100)

    random_x = torch.rand(count.size()[0], count.size()[1], 100).cuda()
    random_y = torch.rand(count.size()[0], count.size()[1], 100).cuda()
    x = torch.where(random_x + random_y > 1.0, 1.0 - random_x, random_x)
    y = torch.where(random_x + random_y > 1.0, 1.0 - random_y, random_y)
    final_x = torch.where(binary_mask, x, torch.Tensor([0.0]).cuda()).unsqueeze(-1).expand(-1, -1, -1,
                                                                                           3)  # (B, n, 100, 3)
    final_y = torch.where(binary_mask, y, torch.Tensor([0.0]).cuda()).unsqueeze(-1).expand(-1, -1, -1,
                                                                                           3)  # (B, n, 100, 3)
    final_a = torch.where(binary_mask, torch.Tensor([1.0]).cuda(), torch.Tensor([0.0]).cuda()).unsqueeze(-1).expand(-1,
                                                                                                                    -1,
                                                                                                                    -1,
                                                                                                                    3)  # (B, n, 100, 3)
    coarse = ver_a * final_a + final_x * edge_ab + final_y * edge_ac
    coarse = coarse.reshape(coarse.size()[0], -1, 3)
    # print("coarse shape:", coarse.shape)
    # print("coarse:", coarse)
    final_coarse1 = torch.Tensor([]).cuda()
    final_coarse2 = torch.Tensor([]).cuda()
    for i in range(coarse.size()[0]):
        non_empty_mask = coarse[i].abs().sum(dim=-1).bool()  # filter out
        indices = non_empty_mask.nonzero(as_tuple=False).squeeze(-1)
        coarse_i = torch.cat((coarse[i][indices], keypoint[i]), 0)
        coarse_i = coarse_i.unsqueeze(0)
        # print("coarse_i:", coarse_i.shape)
        # resample
        idx = np.random.permutation(coarse_i.shape[1])
        if idx.shape[0] < max_n1:
            idx = np.concatenate([idx, np.random.randint(coarse_i.shape[1], size=max_n1 - coarse_i.shape[1])])
        coarse_i_1 = coarse_i[:, idx[:max_n1]]
        if idx.shape[0] < max_n2:
            idx = np.concatenate([idx, np.random.randint(coarse_i.shape[1], size=max_n2 - coarse_i.shape[1])])
        coarse_i_2 = coarse_i[:, idx[:max_n2]]
        final_coarse1 = torch.cat((final_coarse1, coarse_i_1), 0)
        final_coarse2 = torch.cat((final_coarse2, coarse_i_2), 0)
    return final_coarse1, final_coarse2

# concat triangle and skeleton, graph as input
def Batch_Keypoint_graph_interpolation_v2(batch_keypoint, graph, max_n1, max_n2):  # batch_keypoint: (B, k, 3)
    Bn = batch_keypoint.size()[0]
    vertices_index = generate_triangle_vertices_v2(batch_keypoint, graph)  # (B, max_num, 3)
    triangles = batch_keypoint[torch.arange(Bn).unsqueeze(-1).unsqueeze(-1), vertices_index]
    return random_sample_triangle_skeleton(triangles, batch_keypoint, max_n1=max_n1, max_n2=max_n2)

def Batch_Keypoint_graph_interpolation_v3(batch_keypoint, graph, max_n1, max_n2):  # Only Skeleton
    Bn = batch_keypoint.size()[0]
    vertices_index = generate_triangle_vertices_v3(batch_keypoint, graph)  # (B, max_num, 3)
    triangles = batch_keypoint[torch.arange(Bn).unsqueeze(-1).unsqueeze(-1), vertices_index]
    return random_sample_triangle_skeleton(triangles, batch_keypoint, max_n1=max_n1, max_n2=max_n2)

def Batch_Keypoint_graph_interpolation_v4(batch_keypoint, graph):  # batch_keypoint: (B, k, 3)
    Bn = batch_keypoint.size()[0]
    vertices_index = generate_triangle_vertices_v2(batch_keypoint, graph)  # (B, max_num, 3)
    triangles = batch_keypoint[torch.arange(Bn).unsqueeze(-1).unsqueeze(-1), vertices_index]
    return random_sample_triangle_skeleton_v2(triangles, batch_keypoint)

def Batch_Keypoint_graph_interpolation_v5(batch_keypoint, graph, max_n1, max_n2, max_n3):  # batch_keypoint: (B, k, 3)
    Bn = batch_keypoint.size()[0]
    vertices_index = generate_triangle_vertices_v2(batch_keypoint, graph)  # (B, max_num, 3)
    triangles = batch_keypoint[torch.arange(Bn).unsqueeze(-1).unsqueeze(-1), vertices_index]
    return random_sample_triangle_skeleton_v3(triangles, batch_keypoint, max_n1=max_n1, max_n2=max_n2, max_n3=max_n3)

# concat triangle and skeleton
def Batch_Keypoint_graph_interpolation(batch_pc, batch_keypoint, max_n):  # batch_keypoint: (B, k, 3)
    Bn = batch_keypoint.size()[0]
    graph = init_graph(batch_pc, batch_keypoint)
    vertices_index = generate_triangle_vertices_v2(batch_keypoint, graph)  # (B, max_num, 3)
    triangles = batch_keypoint[torch.arange(Bn).unsqueeze(-1).unsqueeze(-1), vertices_index]
    return random_sample_triangle_skeleton(triangles, batch_keypoint, max_n=max_n)


def random_sample_zero_padding(triangle, keypoint, interval=0.00005, max_n=1024):  # (B, n, 3, 3)
    ver_a = triangle[..., 0, :].unsqueeze(2).expand(-1, -1, 100, -1)
    edge_ab = (triangle[..., 1, :] - triangle[..., 0, :]).unsqueeze(2).expand(-1, -1, 100, -1)
    edge_ac = (triangle[..., 2, :] - triangle[..., 0, :]).unsqueeze(2).expand(-1, -1, 100, -1)

    area, side_ab, side_bc, side_ac = Batch_calculate_area(triangle)
    area = (area / interval).long()
    count = torch.clamp(area, 0, 100)  # (B, n)
    count = count.unsqueeze(-1)  # (B, n, 1)

    base_index = torch.arange(100).expand(count.size()[0], count.size()[1], 100).cuda()
    binary_mask = base_index < count  # (B, n, 100)

    random_x = torch.rand(count.size()[0], count.size()[1], 100).cuda()
    random_y = torch.rand(count.size()[0], count.size()[1], 100).cuda()
    x = torch.where(random_x + random_y > 1.0, 1.0 - random_x, random_x)
    y = torch.where(random_x + random_y > 1.0, 1.0 - random_y, random_y)
    final_x = torch.where(binary_mask, x, torch.Tensor([0.0]).cuda()).unsqueeze(-1).expand(-1, -1, -1,
                                                                                           3)  # (B, n, 100, 3)
    final_y = torch.where(binary_mask, y, torch.Tensor([0.0]).cuda()).unsqueeze(-1).expand(-1, -1, -1,
                                                                                           3)  # (B, n, 100, 3)
    final_a = torch.where(binary_mask, torch.Tensor([1.0]).cuda(), torch.Tensor([0.0]).cuda()).unsqueeze(-1).expand(-1,
                                                                                                                    -1,
                                                                                                                    -1,
                                                                                                                    3)  # (B, n, 100, 3)
    coarse = ver_a * final_a + final_x * edge_ab + final_y * edge_ac
    coarse = coarse.reshape(coarse.size()[0], -1, 3)
    # print("coarse shape:", coarse.shape)
    # print("coarse:", coarse)
    final_coarse = torch.Tensor([]).cuda()
    for i in range(coarse.size()[0]):
        non_empty_mask = coarse[i].abs().sum(dim=-1).bool()  # filter out
        indices = non_empty_mask.nonzero(as_tuple=False).squeeze(-1)
        coarse_i = torch.cat((coarse[i][indices], keypoint[i]), 0)
        coarse_i = coarse_i.unsqueeze(0)
        # print(coarse_i.shape)
        # resample
        idx = np.random.permutation(coarse_i.shape[1])
        if idx.shape[0] < max_n:
            idx = np.concatenate([idx, np.random.randint(coarse_i.shape[1], size=max_n - coarse_i.shape[1])])
        coarse_i = coarse_i[:, idx[:max_n]]
        final_coarse = torch.cat((final_coarse, coarse_i), 0)
    return final_coarse


# generate triangular interpolation from keypoints(using Batch_EuclideanDistances)
def Batch_Keypoint_random_sample(batch_keypoint, max_n):  # batch_keypoint: (B, n, 3)
    batchresults = torch.Tensor([]).cuda()
    Dis_mat = Batch_EuclideanDistances(batch_keypoint.double(), batch_keypoint.double())  # (B, n, n)
    # print(torch.sum(torch.isnan(Dis_mat)))
    vertices = torch.topk(Dis_mat, 3, dim=-1, largest=False).indices  # closest vertices:(B, n, 3)\
    # print(torch.sum(torch.isnan(vertices)))
    # print(vertices.shape)
    closest_keypoints = Batch_index_select(batch_keypoint, 2, vertices)  # (B, n, 3, 3)
    # print(torch.sum(torch.isnan(closest_keypoints)))
    return random_sample_zero_padding(closest_keypoints, batch_keypoint, max_n=max_n)

def Batch_interp_triangle_v2(closest_keypoints, interval=0.01):  # closest_keypoints (B, n, 3, 3)
    device = closest_keypoints.device
    area, side_ab, side_bc, side_ac = Batch_calculate_area(closest_keypoints)
    height = torch.div(area * 2, side_bc)
    height = (height / interval).long()
    count = torch.clamp(height, min=1, max=10)
    # print(count)
    # to be modified !!!
    coarse = torch.Tensor([]).cuda()
    for i in range(count.size()[0]):
        coarse_i = torch.Tensor([]).cuda()
        for j in range(count.size()[1]):
            a, b, c = closest_keypoints[i][j][0], closest_keypoints[i][j][1], closest_keypoints[i][j][2]
            interp_x = torch.linspace(0.0, 1.0, count[i][j]).unsqueeze(0).unsqueeze(-1).to(device)
            interp_y = 1.0 - interp_x
            ab = a * interp_x + b * interp_y
            ac = a * interp_x + c * interp_y
            # print(ab.shape)  # (1, n, 3)
            result = torch.cat((ab, ac), 1)

            # for k in range(0, count[i][j] - 1):
            #     ab, ac = ab.squeeze(0), ac.squeeze(0)
            #     s, e = ab[k], ac[k]
            #     side_k = ((s[0] - e[0]) ** 2 + (s[1] - e[1]) ** 2 + (s[2] - e[2]) ** 2 + 1e-10) ** 0.5
            #     # print("side_k: ", side_k)
            #     count_k = max(1, min(10, int((side_k / interval).item())))
            #     interp_u = torch.linspace(0.0, 1.0, count_k).unsqueeze(0).unsqueeze(-1).to(device)
            #     interp_v = 1 - interp_u
            #     se = s * interp_u + e * interp_v
            #     # print(se.shape)
            #     result = torch.cat((result, se), 1)
            #     # print("now result: ", result.shape)

            coarse_i = torch.cat((coarse_i, result), 1)
            # print(coarse_i.shape)
        # print(result.shape)
        coarse = torch.cat((coarse, coarse_i), 0)
        # print(coarse.shape)
    return coarse


def single_sample_interpolation(pc, keypoint):  # pc(N,3), kp(K,3)
    Bn = 1
    batch_pc = pc.unsqueeze(0)
    batch_keypoint = keypoint.unsqueeze(0)
    graph = init_graph(batch_pc, batch_keypoint)
    vertices_index = generate_triangle_vertices_v2(batch_keypoint, graph)  # (B, max_num, 3)
    triangles = batch_keypoint[torch.arange(Bn).unsqueeze(-1).unsqueeze(-1), vertices_index]
    results = Batch_interp_triangle_v2(triangles, 0.01)
    return results.squeeze(0)

if __name__ == '__main__':

    '''runtime test'''
    pc = torch.randn(32, 1024, 3).cuda()
    kp = torch.randn(32, 16, 3).cuda()
    # Batch_Keypoint_interpolation(test, 1024)
    begin_time = time()
    print(Batch_Keypoint_graph_interpolation_v2(pc, kp, 512).shape)
    # print(Batch_Keypoint_interpolation(test, 1024).shape)
    # print(Batch_Keypoint_random_sample(pc, 512).shape)
    end_time = time()
    print("calculate time: ", end_time - begin_time)

import logging
import os
import sys
import importlib
import argparse
import munch
import yaml
from utils.vis_utils import plot_single_pcd, save_pointcloud_ply
from utils.train_utils import *
from utils.triangle_utils import *
from dataset_old import ShapeNetH5, ShapeNetPcd
from collections import OrderedDict
import torch
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(proj_dir, "utils/Pointnet2.PyTorch/pointnet2/"))
import pointnet2_utils as pn2

def fps_d(point, num):
    # b, N, D
    pid_fps = pn2.furthest_point_sample(point, num)
    x_fps = pn2.gather_operation(point.transpose(1, 2).contiguous(), pid_fps).transpose(1, 2).contiguous()

    return x_fps # b, Nf, D


def test():
    dataset_test = ShapeNetPcd(test=True, npoints=args.num_points, shuffle=False)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                  shuffle=False, num_workers=int(args.workers))
    dataset_length = len(dataset_test)
    logging.info('Length of test dataset:%d', len(dataset_test))

    # load model
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(model_module.Model(args, up_factors=[1,2])).cuda()
        net.load_state_dict(torch.load(args.load_model)['net_state_dict'], strict=False)
    else:
        net = model_module.Model(args, up_factors=[1,2]).cuda()
        state_dict = torch.load(args.load_model)['net_state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name]=v
        net.load_state_dict(new_state_dict, strict=False)
    
    logging.info("%s's previous weights loaded." % args.model_name)
    net.eval()

    metrics = ['cd_p', 'cd_t', 'f1']
    test_loss_meters = {m: AverageValueMeter() for m in metrics}
    test_loss_cat = torch.zeros([16, 4], dtype=torch.float32).cuda()
    cat_num = torch.ones([16, 1], dtype=torch.float32).cuda() * 150
    cat_name = ['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'vessel']
    idx_to_plot = [i for i in range(0, 1200, 1)]

    logging.info('Testing...')
    if args.save_vis:
        save_gt_path = os.path.join(log_dir, 'pics', 'gt')
        save_partial_path = os.path.join(log_dir, 'pics', 'partial')
        save_completion_path = os.path.join(log_dir, 'pics', 'completion')
        os.makedirs(save_gt_path, exist_ok=True)
        os.makedirs(save_partial_path, exist_ok=True)
        os.makedirs(save_completion_path, exist_ok=True)

    if args.save_ply:
        save_ply_path = os.path.join(log_dir, 'fps')
        os.makedirs(save_ply_path, exist_ok=True)
    with torch.no_grad():
        for i, data in enumerate(dataloader_test, 567):
            
            label, inputs_cpu, gt_cpu, gt_2048 = data
            gt_2048 = gt_2048.float().cuda()
            inputs = inputs_cpu.float().cuda()
            gt = gt_cpu.float().cuda()
            result_dict = net(inputs, gt, gt_2048, label, epochidx=0, is_training=False)
            

            for k, v in test_loss_meters.items():
                v.update(result_dict[k].mean().item())

            for j, l in enumerate(label):
                for ind, m in enumerate(metrics):
                    test_loss_cat[int(l), ind] = result_dict[m][int(j)]

            logging.info('test [%d/%d]' % (i, dataset_length))

            ply_gt = 'object_%d_gt.ply' % i
            ply_ckp1 = 'object_%d_kp1.ply' % i
            ply_kpc1 = 'object_%d_kpc1.ply' % i
            ply_ckp2 = 'object_%d_kp2.ply' % i
            ply_kpc2 = 'object_%d_kpc2.ply' % i
            ply_ckp3 = 'object_%d_kp3.ply' % i
            ply_kpc3 = 'object_%d_kpc3.ply' % i

            # kp1 = result_dict['kp1'][j]
            # kp2 = result_dict['kp2'][j]
            # kp3 = result_dict['kp3'][j]

            # kpc1 = single_sample_interpolation(gt[j], kp1).unsqueeze(0)
            # kpc2 = single_sample_interpolation(gt[j], kp2).unsqueeze(0)
            # kpc3 = single_sample_interpolation(gt[j], kp3).unsqueeze(0)
            save_pointcloud_ply(fps_d(gt_2048, 32), os.path.join(save_ply_path, ply_ckp1))
            # save_pointcloud_ply(kpc1, os.path.join(save_ply_path, ply_kpc1))
            # save_pointcloud_ply(result_dict['kp2'][j].unsqueeze(0), os.path.join(save_ply_path, ply_ckp2))
            # save_pointcloud_ply(kpc2, os.path.join(save_ply_path, ply_kpc2))
            # save_pointcloud_ply(result_dict['kp3'][j].unsqueeze(0), os.path.join(save_ply_path, ply_ckp3))
            # save_pointcloud_ply(kpc3, os.path.join(save_ply_path, ply_kpc3))
            
            save_pointcloud_ply(gt, os.path.join(save_ply_path, ply_gt))

                # for j in range(args.batch_size):
                #     idx = i * args.batch_size + j
                #     taxonomy_id, model_id = name[j].split('/')[0], name[j].split('/')[1]
                #     if idx in idx_to_plot:
                #         ply_gt = '%s_%s_gt.ply' % (taxonomy_id, model_id)
                #         ply_input = '%s_%s_input.ply' % (taxonomy_id, model_id)
                #         ply_ouput = '%s_%s.ply' % (taxonomy_id, model_id)

                #         save_pointcloud_ply(gt[j].unsqueeze(0), os.path.join(save_ply_path, ply_gt))
                #         save_pointcloud_ply(inputs[j].unsqueeze(0), os.path.join(save_ply_path, ply_input))
                #         save_pointcloud_ply(result_dict['out2'][j].unsqueeze(0), os.path.join(save_ply_path, ply_ouput))

        logging.info('Loss per category:')
        category_log = ''
        for i in range(8):
            category_log += 'category name: %s' % (cat_name[i])
            for ind, m in enumerate(metrics):
                scale_factor = 1 if m == 'f1' else 10000
                category_log += '%s: %f' % (m, test_loss_cat[i, 0] / cat_num[i] * scale_factor)
        logging.info(category_log)

        logging.info('Overview results:')
        overview_log = ''
        for metric, meter in test_loss_meters.items():
            overview_log += '%s: %f ' % (metric, meter.avg)
        logging.info(overview_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))

    if not args.load_model:
        raise ValueError('Model path must be provided to load model!')

    exp_name = os.path.basename(args.load_model)
    log_dir = os.path.dirname(args.load_model)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'test.log')),
                                                      logging.StreamHandler(sys.stdout)])

    test()

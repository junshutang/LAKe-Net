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
from utils.average_meter import AverageMeter
from utils.metrics import Metrics
from dataset_old import ShapeNetH5, ShapeNetPcd
import torch
from collections import OrderedDict
# from Chamfer3D.dist_chamfer_3D import chamfer_3DDist

# chamfer_dist = chamfer_3DDist()
# def chamfer(p1, p2):
#     d1, d2, _, _ = chamfer_dist(p1, p2)
#     return torch.mean(d1) + torch.mean(d2)

# def chamfer_sqrt(p1, p2):
#     d1, d2, _, _ = chamfer_dist(p1, p2)
#     d1 = torch.mean(torch.sqrt(d1))
#     d2 = torch.mean(torch.sqrt(d2))
#     return (d1 + d2) / 2

def test():
    dataset_test = ShapeNetPcd(test=True, npoints=args.num_points, shuffle=False)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                  shuffle=False, num_workers=int(args.workers))
    dataset_length = len(dataset_test)
    logging.info('Length of test dataset:%d', len(dataset_test))

    # load model
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(model_module.Model(args, up_factors=[2,4])).cuda()
        net.load_state_dict(torch.load(args.load_model)['net_state_dict'], strict=False)
    else:
        net = model_module.Model(args, up_factors=[2,4]).cuda()
        state_dict = torch.load(args.load_model)['net_state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name]=v
        net.load_state_dict(new_state_dict, strict=False)
    
    logging.info("%s's previous weights loaded." % args.model_name)
    net.eval()
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()

    logging.info('Testing...')

    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            
            label, name, inputs_cpu, gt_cpu, gt_2048 = data
            inputs = inputs_cpu.float().cuda()
            gt = gt_cpu.float().cuda()
            gt_2048 = gt_2048.float().cuda()
            result_dict = net(inputs, gt, gt_2048, label, epochidx=0, is_training=False)
            data = result_dict['out2']
            # cd = chamfer_sqrt(data.reshape(-1, 16384, 3).contiguous(), gt.reshape(-1, 16384, 3).contiguous()).item() * 1e3
            # _metrics = [cd]
            _metrics = Metrics.get(data.reshape(-1, 16384, 3), gt.reshape(-1, 16384, 3).contiguous())
            test_metrics.update(_metrics)
            
            taxonomy_id, model_id = name[0].split('/')[0], name[0].split('/')[1]
            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)

            print('Test[%d/%d] Taxonomy = %s Sample = %s Metrics = %s' %
                         (i + 1, len(dataloader_test), taxonomy_id, model_id, ['%.4f' % m for m in _metrics]))

    # Print testing results
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    for metric in test_metrics.items:
        print(metric, end='\t')
    print()

    for taxonomy_id in category_metrics:
        print(taxonomy_id, end='\t')
        print(category_metrics[taxonomy_id].count(0), end='\t')
        for value in category_metrics[taxonomy_id].avg():
            print('%.4f' % value, end='\t')
        print()

    print('Overall', end='\t\t\t')
    for value in test_metrics.avg():
        print('%.4f' % value, end='\t')
    print('\n')


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

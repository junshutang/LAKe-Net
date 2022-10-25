import os
import sys
import math
import importlib
import datetime
import random
import munch
import yaml
import argparse
import torch.optim as optim
import torch
from utils.train_utils import *
import logging
from dataset_old import ShapeNetPcd

def train():
    logging.info(str(args))
    metrics = ['cd_p', 'cd_t', 'f1']
    best_pre_epoch_losses = {m: (0, 0) if m == 'f1' else (0, math.inf) for m in metrics}
    best_epoch_losses = {m: (0, 0) if m == 'f1' else (0, math.inf) for m in metrics}
    train_loss_meter = AverageValueMeter()
    pre_val_loss_meters = {m: AverageValueMeter() for m in metrics}
    val_loss_meters = {m: AverageValueMeter() for m in metrics}

    dataset = ShapeNetPcd(train=True, npoints=args.num_points)
    dataset_val = ShapeNetPcd(val=True, npoints=args.num_points)
    dataloader_pre = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=int(args.workers))
    dataloader_pre_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=int(args.workers))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=int(args.workers))
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=int(args.workers))
    logging.info('Length of train dataset:%d', len(dataset))
    logging.info('Length of val dataset:%d', len(dataset_val))

    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)

    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(model_module.Model(args, up_factors=[1,2]))
    else:
        net = model_module.Model(args, up_factors=[1,2])

    net.cuda()
    net_d = None

    lr = args.lr
    if args.lr_decay:
        if args.lr_decay_interval and args.lr_step_decay_epochs:
            raise ValueError('lr_decay_interval and lr_step_decay_epochs are mutually exclusive!')
        if args.lr_step_decay_epochs:
            decay_epoch_list = [int(ep.strip()) for ep in args.lr_step_decay_epochs.split(',')]
            decay_rate_list = [float(rt.strip()) for rt in args.lr_step_decay_rates.split(',')]

    optimizer = getattr(optim, args.optimizer)
    if args.optimizer == 'Adagrad':
        if torch.cuda.device_count() > 1:
            optimizer = optimizer(net.module.parameters(), lr=lr, initial_accumulator_value=args.initial_accum_val)
        else:
            optimizer = optimizer(net.parameters(), lr=lr, initial_accumulator_value=args.initial_accum_val)
    else:
        betas = args.betas.split(',')
        betas = (float(betas[0].strip()), float(betas[1].strip()))
        if torch.cuda.device_count() > 1:
            optimizer = optimizer(net.module.parameters(), lr=lr, weight_decay=args.weight_decay, betas=betas)
        else:
            optimizer = optimizer(net.parameters(), lr=lr, weight_decay=args.weight_decay, betas=betas)

    alpha = None
    if args.varying_constant:
        varying_constant_epochs = [int(ep.strip()) for ep in args.varying_constant_epochs.split(',')]
        varying_constant = [float(c.strip()) for c in args.varying_constant.split(',')]
        assert len(varying_constant) == len(varying_constant_epochs) + 1

    if args.load_model:
        ckpt = torch.load(args.load_model)
        if torch.cuda.device_count() > 1:
            net.load_state_dict(ckpt['net_state_dict'], strict=False)
        else:
            net.load_state_dict(ckpt['net_state_dict'])
        logging.info("%s's previous weights loaded." % args.model_name)

    for epoch in range(args.start_epoch, args.nepoch):
        train_loss_meter.reset()
        if torch.cuda.device_count() > 1:
            net.module.train()
        else:
            net.train()

        if args.varying_constant:
            for ind, ep in enumerate(varying_constant_epochs):
                if epoch < args.pretrain_epoch and epoch < ep:
                    alpha = varying_constant[ind]
                    break
                elif epoch < args.pretrain_epoch and ind == len(varying_constant_epochs)-1 and epoch >= ep:
                    alpha = varying_constant[ind+1]
                    break
                elif epoch >= args.pretrain_epoch and epoch < args.pretrain_epoch + ep:
                    alpha = varying_constant[ind]
                    break
                elif epoch >= args.pretrain_epoch and ind == len(varying_constant_epochs)-1 and epoch >= args.pretrain_epoch + ep:
                    alpha = varying_constant[ind+1]
                    break

        if args.lr_decay:
            if args.lr_decay_interval:
                if epoch == args.pretrain_epoch :
                    lr = args.lr
                elif epoch > 0 and epoch % args.lr_decay_interval == 0:
                    lr = lr * args.lr_decay_rate
            if args.lr_clip:
                lr = max(lr, args.lr_clip)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        ######################################
        # pre-train keypoint predict network
        ######################################
        if epoch < args.pretrain_epoch:
            logging.info('######### pre-training #########')
            
            for i, data in enumerate(dataloader_pre, 0):

                label, inputs, gt, gt_2048 = data
                inputs = inputs.float().cuda()
                gt = gt.float().cuda()
                label = label.cuda()
                gt_2048 = gt_2048.cuda()
                
                optimizer.zero_grad()
                cout1, cout2, closs1, closs2, loss_kp, loss_cls, net_loss = net(inputs, gt, gt_2048, label, epochidx=epoch, alpha=alpha) # Add epoch
                # loss_kp, loss_cls, net_loss = net(inputs, gt, label, epochidx=epoch, alpha=alpha) # Add epoch
                train_loss_meter.update(net_loss.mean().item())
                
                if torch.cuda.device_count() > 1:
                    net_loss.backward(torch.ones(torch.cuda.device_count()).cuda())
                else:
                    net_loss.backward()
                    
                optimizer.step()

                if i % args.step_interval_to_print == 0:
                    logging.info(exp_name + ' train [%d: %d/%d] coarse_loss: %f, fine_loss: %f, loss_kp: %f, loss_cls: %f, total_loss: %f lr: %f' %
                                 (epoch, i, len(dataset) / (args.batch_size), closs1.mean().item(), closs2.mean().item(), loss_kp.mean().item(), loss_cls.mean().item(), net_loss.mean().item(), lr) + ' alpha: ' + str(alpha))
                
        ######################################
        # train completion network 
        ######################################
        # Load pretrained model
        elif epoch >= args.pretrain_epoch:

            if epoch == args.pretrain_epoch:
                logging.info('######### Load pretrained model #########')
                ckpt = torch.load("%s/best_pre_cd_t_network.pth" % log_dir)
                net.load_state_dict(ckpt['net_state_dict'], strict=False)
                logging.info("Previous pretrained weights loaded.")

                for name, parms in net.module.named_parameters():  
                    if 'kp_detect' in name or 'encoder' in name:
                        print(name)
                        parms.requires_grad=False
                logging.info("Fixed pretrained keypoint predictor and encoder.")

            logging.info('######### completion training #########')


            for i, data in enumerate(dataloader, 0):
                # if i == 3:
                #     break
                label, inputs, gt, gt_2048 = data
                inputs = inputs.float().cuda()
                gt = gt.float().cuda()
                label = label.cuda()
                gt_2048 = gt_2048.cuda()

                optimizer.zero_grad()
                pout1, pout2, ploss1, ploss2, kp_com_loss, kp_loss, kp_coarse_loss, loss_feat, net_loss = net(inputs, gt, gt_2048, label, epochidx=epoch, alpha=alpha) # Add epoch
                
                train_loss_meter.update(net_loss.mean().item())
                
                if torch.cuda.device_count() > 1:
                    net_loss.backward(torch.ones(torch.cuda.device_count()).cuda())
                else:
                    net_loss.backward()

                optimizer.step()

                if i % args.step_interval_to_print == 0:
                    logging.info(exp_name + ' train [%d: %d/%d] coarse_loss: %f, fine_loss: %f, kp_com_loss: %f, kp_loss: %f, kp_coarse_loss: %f, feat_loss: %f, total_loss: %f lr: %f' % \
                                 (epoch, i, len(dataset) / args.batch_size, ploss1.mean().item(), ploss2.mean().item(), \
                                   kp_com_loss.mean().item(), kp_loss.mean().item(), kp_coarse_loss.mean().item(), \
                                   loss_feat.mean().item(), net_loss.mean().item(), lr) + ' alpha: ' + str(alpha))

        if epoch % args.epoch_interval_to_save == 0 and epoch < args.pretrain_epoch:
            save_model('%s/pre-network.pth' % log_dir, net, net_d=net_d)
            logging.info("Saving pretrained net...")
        elif epoch % args.epoch_interval_to_save == 0 and epoch >= args.pretrain_epoch:
            save_model('%s/network.pth' % log_dir, net, net_d=net_d)
            logging.info("Saving net...")

        if epoch % args.epoch_interval_to_val == 0 or epoch == args.nepoch - 1:
            val(net, epoch, pre_val_loss_meters, val_loss_meters, dataloader_val, dataloader_pre_val, best_pre_epoch_losses, best_epoch_losses)


def val(net, epoch, pre_val_loss_meters, val_loss_meters, dataloader_val, dataloader_pre_val, best_pre_epoch_losses, best_epoch_losses):
    logging.info('Testing...')

    for pre_v in pre_val_loss_meters.values():
        pre_v.reset()
    for v in val_loss_meters.values():
        v.reset()
    
    if torch.cuda.device_count() > 1:
        net.module.eval()
    else:
        net.eval()

    with torch.no_grad():

        if epoch < args.pretrain_epoch:
            logging.info('######### pre-trained-testing #########')
            for i, data in enumerate(dataloader_pre_val):
                label, inputs, gt, gt_2048 = data
                inputs = inputs.float().cuda()
                gt = gt.float().cuda()
                label = label.cuda()

                pretrain_result_dict = net(inputs, gt, gt_2048, label, is_training=False, epochidx=epoch, logging=logging)
                for k, v in pre_val_loss_meters.items():
                    v.update(pretrain_result_dict[k].mean().item())

        elif epoch >= args.pretrain_epoch:
            logging.info('######### completion-testing #########')
            for i, data in enumerate(dataloader_val):
                label, inputs, gt, gt_2048 = data
                inputs = inputs.float().cuda()
                gt = gt.float().cuda()
                label = label.cuda()

                result_dict = net(inputs, gt, gt_2048, label, is_training=False, epochidx=epoch, logging=logging)
                for k, v in val_loss_meters.items():
                    v.update(result_dict[k].mean().item())

        if epoch < args.pretrain_epoch:
            logging.info('######### pre-training-eval #########')
            fmt = 'best_pre_%s: %f [epoch %d]; '
            best_log = ''
            for loss_type, (curr_best_epoch, curr_best_loss) in best_pre_epoch_losses.items():
                if (pre_val_loss_meters[loss_type].avg < curr_best_loss and loss_type != 'f1') or \
                        (pre_val_loss_meters[loss_type].avg > curr_best_loss and loss_type == 'f1'):
                    best_pre_epoch_losses[loss_type] = (epoch, pre_val_loss_meters[loss_type].avg)
                    save_model('%s/best_pre_%s_network.pth' % (log_dir, loss_type), net)
                    logging.info('Best pretrained %s net saved!' % loss_type)
                    best_log += fmt % (loss_type, best_pre_epoch_losses[loss_type][1], best_pre_epoch_losses[loss_type][0])
                else:
                    best_log += fmt % (loss_type, curr_best_loss, curr_best_epoch)

            curr_log = ''
            for loss_type, meter in pre_val_loss_meters.items():
                curr_log += 'curr_pre_%s: %f; ' % (loss_type, meter.avg)

            logging.info(curr_log)
            logging.info(best_log)

        elif epoch >= args.pretrain_epoch:
            logging.info('######### completion evaluation #########')
            fmt = 'best_%s: %f [epoch %d]; '
            best_log = ''
            for loss_type, (curr_best_epoch, curr_best_loss) in best_epoch_losses.items():
                if (val_loss_meters[loss_type].avg < curr_best_loss and loss_type != 'f1') or \
                        (val_loss_meters[loss_type].avg > curr_best_loss and loss_type == 'f1'):
                    best_epoch_losses[loss_type] = (epoch, val_loss_meters[loss_type].avg)
                    save_model('%s/best_%s_network.pth' % (log_dir, loss_type), net)
                    logging.info('Best %s net saved!' % loss_type)
                    best_log += fmt % (loss_type, best_epoch_losses[loss_type][1], best_epoch_losses[loss_type][0])
                else:
                    best_log += fmt % (loss_type, curr_best_loss, curr_best_epoch)

            curr_log = ''
            for loss_type, meter in val_loss_meters.items():
                curr_log += 'curr_%s: %f; ' % (loss_type, meter.avg)

            logging.info(curr_log)
            logging.info(best_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    parser.add_argument('-n', '--nolog', action='store_true', default=False, help='not save log file')
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))
    print("GPU Number:", torch.cuda.device_count(), "GPUs!")
 
    time = datetime.datetime.now().isoformat()[:19]
    if args.load_model:
        exp_name = os.path.basename(os.path.dirname(args.load_model))
        log_dir = os.path.dirname(args.load_model)
    else:
        exp_name = args.model_name + '_' + args.loss + '_' + args.flag + '_' + time
        log_dir = os.path.join(args.work_dir, exp_name)
    print(log_dir)       
    if arg.nolog:
        logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    else:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'train.log')),
                                                   logging.StreamHandler(sys.stdout)])
        os.system('cp ./cfgs/'+ args.model_name +'.yaml %s' % log_dir)
        os.system('cp ./train_pre.py %s' % log_dir)
        os.system('cp ./test.py %s' % log_dir)
        os.system('cp ./models/'+ args.model_name +'.py %s' % log_dir)

    train()




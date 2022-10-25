import os
import open3d as o3d
import random
import torch
import numpy as np
import torch.utils.data as data
import h5py
import munch
import yaml
import argparse

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]

class ShapeNetPcd(data.Dataset): 
    def __init__(self, train=False, val=False, test=False, viz=False, npoints=2048, shuffle=True):
        super(ShapeNetPcd, self).__init__()
        if train:
            self.list_path = './data/train.list'
        elif val:
            self.list_path = './data/val.list'
        elif test:
            self.list_path = './data/test.list'
        elif viz:
            self.list_path = './data/viz.list'
        else:
            print("ERROR INPUT")
        self.npoints = npoints
        self.train = train
        self.val = val
        self.test = test
        self.viz = viz
        self.cate_path = './data/synsetoffset2category.txt'
        self.cate_dict = {}
        lable_index = 0
        file = open(self.cate_path, 'r')
        for line in file.readlines():
            line = line.strip()
            k = line.split('\t')[1]
            self.cate_dict[k] = lable_index
            lable_index += 1
        file.close()
        with open(os.path.join(self.list_path)) as file:
            self.model_list = [line.strip() for line in file]
        if shuffle:
            random.shuffle(self.model_list)
        self.len = len(self.model_list)

    def __getitem__(self, index):
        model_id = self.model_list[index]
        rand_id = random.randint(0,7)
        def read_pcd(filename):
            pcd = o3d.io.read_point_cloud(filename)
            return torch.from_numpy(np.array(pcd.points)).float()
        if self.train:
            complete = read_pcd(os.path.join("./data/train/complete/", model_id + '.pcd'))
            partial = read_pcd(os.path.join("./data/train/partial/", model_id + '/0%d.pcd' % rand_id))
        elif self.val:
            complete = read_pcd(os.path.join("./data/val/complete/", model_id + '.pcd'))
            partial = read_pcd(os.path.join("./data/val/partial/", model_id + '/00.pcd'))
        elif self.test:
            complete = read_pcd(os.path.join("./data/test/complete/", model_id + '.pcd'))
            partial = read_pcd(os.path.join("./data/test/partial/", model_id + '/00.pcd'))
        elif self.viz:
            complete = read_pcd(os.path.join("./data/train/complete/", model_id + '.pcd'))
            partial = read_pcd(os.path.join("./data/train/partial/", model_id + '/0%d.pcd' % rand_id))
        else:
            print("ERROR INPUT")
               
        return self.cate_dict[model_id.split('/')[0]], resample_pcd(partial, 2048), resample_pcd(complete, self.npoints), resample_pcd(complete, 2048)

    def __len__(self):
        return self.len


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    print('test')
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))
    
    data = ShapeNetPcd(test=True, npoints=16384)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=int(args.workers))
    for label, partial, complete in DataLoader:
        print(label.shape)
        print(partial.shape)
        print(complete.shape)
        break
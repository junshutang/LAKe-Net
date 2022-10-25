import os
import open3d as o3d
import random
import numpy as np
import h5py
import munch
import json
import yaml
import argparse
import torch
import torch.utils.data as data
import torch.nn.functional as F

category_dict_55 = {"02691156": 0, 
                 "02747177": 1, 
                 "02773838": 2, 
                 "02801938": 3, 
                 "02808440": 4, 
                 "02818832": 5, 
                 "02828884": 6, 
                 "02843684": 7, 
                 "02871439": 8, 
                 "02876657": 9, 
                 "02880940": 10, 
                 "02924116": 11, 
                 "02933112": 12, 
                 "02942699": 13, 
                 "02946921": 14, 
                 "02954340": 15, 
                 "02958343": 16, 
                 "02992529": 17, 
                 "03001627": 18, 
                 "03046257": 19, 
                 "03085013": 20, 
                 "03207941": 21, 
                 "03211117": 22, 
                 "03261776": 23, 
                 "03325088": 24, 
                 "03337140": 25, 
                 "03467517": 26, 
                 "03513137": 27, 
                 "03593526": 28, 
                 "03624134": 29, 
                 "03636649": 30, 
                 "03642806": 31, 
                 "03691459": 32, 
                 "03710193": 33, 
                 "03759954": 34, 
                 "03761084": 35, 
                 "03790512": 36, 
                 "03797390": 37, 
                 "03928116": 38, 
                 "03938244": 39, 
                 "03948459": 40, 
                 "03991062": 41, 
                 "04004475": 42, 
                 "04074963": 43, 
                 "04090263": 44, 
                 "04099429": 45, 
                 "04225987": 46, 
                 "04256520": 47, 
                 "04330267": 48, 
                 "04379243": 49, 
                 "04401088": 50, 
                 "04460130": 51, 
                 "04468005": 52, 
                 "04530566": 53, 
                 "04554684": 54}


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data


def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]

class ShapeNet55(data.Dataset):
    def __init__(self, train=True, npoints=2048, train_list='./shapenet55/ShapeNet-55/train.txt', val_list='./shapenet55/ShapeNet-55/test_small.txt'):
        if train:
            self.list_path = train_list
        else:
            self.list_path = val_list
        self.npoints = npoints
        self.train = train
        self.data_path = './shapenet55/shapenet_pc/'
        self.cate_path = './shapenet55/shapenet_synset_dict.json'
        with open(self.cate_path,'r') as load_f:
            self.cate_dict = json.load(load_f)
        with open(self.list_path, 'r') as f:
            lines = f.readlines()
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print(f'[DATASET] {len(self.file_list)} instances were loaded')
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = np.load(os.path.join(self.data_path, sample['file_path'])).astype(np.float32)
        data = self.pc_norm(data)
        complete = torch.from_numpy(data).float()

        return category_dict_55[sample['taxonomy_id']], sample['taxonomy_id'], complete, resample_pcd(complete, 2048)

    def __len__(self):
        return len(self.file_list)


if __name__ == '__main__':

    # pc = torch.randn(3, 3)
    # print(pc)
    # print(resample_pcd(pc, 5))
    
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    print('test')
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))
    #
    data = ShapeNet55(train=True, npoints=8192)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=int(args.workers))
    for label, complete in DataLoader:
        print(label)
        print(complete.shape)
        break
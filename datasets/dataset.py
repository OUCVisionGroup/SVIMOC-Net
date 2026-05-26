import itertools

import cv2
from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np
import h5py
import math

def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


class Crowd(data.Dataset):
    def __init__(self, root, crop_size, downsample_ratio, method='train', info=None):
        self.im_list = sorted(glob(os.path.join(root, 'images/*.jpg')))
        if method not in ['train', 'val', 'test']:
            raise Exception('Method is not implemented!')
        self.label_list = []
        if method == 'train':
            try:
                with open(info) as f:
                    for i in f:
                        self.label_list.append(i.strip())
            except:
                raise Exception("please give right info")

            labeled = []
            for i in self.im_list:
                if os.path.basename(i) in self.label_list:
                    labeled.append(1)
                else:
                    labeled.append(0)
            labeled = np.array(labeled)
            self.labeled_idx = np.where(labeled == 1)[0]
            self.unlabeled_idx = np.where(labeled == 0)[0]

        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        self.root = root
        self.method = method
        assert self.c_size % self.d_ratio == 0
        self.w_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.png_transform = transforms.Compose([
                            transforms.Grayscale(num_output_channels=1),  
                            transforms.ToTensor(), 
        ])
        self.s_transform = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):

        im_path = self.im_list[item]
        dir_path = os.path.dirname(im_path)
        name = os.path.basename(im_path).split('.')[0]

        index = int(name)
         
        base = ((index - 1) // 10) * 10 + 1
        base2 = ((index - 1) // 10) * 10 + 10

        prev_index = int(max(base,index-1))
        post_index = int(min(base2,index+1))
        
        # gd_path = os.path.join(self.root, 'gt_points', '{}.npy'.format(name))

        prev_img_path = os.path.join(dir_path,'%03d.jpg'%(prev_index))
        img = Image.open(im_path).convert('RGB')
        post_img_path = os.path.join(dir_path,'%03d.jpg'%(post_index))
        prev_img = Image.open(prev_img_path).convert('RGB')
        post_img = Image.open(post_img_path).convert('RGB')

        # png_path = os.path.join(self.root, 'images2', '{}.png'.format(name))
        # png = Image.open(png_path).convert('RGB')
        
    
        if self.method == 'train':
            png_path = os.path.join(self.root, 'images', '{}.png'.format(name))
            png = Image.open(png_path).convert('RGB')
            if index  % 10 == 1:
                den_map_path = os.path.join(self.root, 'gt_den', '{}.h5'.format(name))
                den_map = h5py.File(den_map_path, 'r')['density_map']
               
            else:
                # Create an empty density map if the condition is not met
                den_map = np.zeros((1080, 1920)) 
            label = (os.path.basename(im_path) in self.label_list)
            
            return self.train_transform_density_map3(prev_img, img, png, post_img, den_map, label)

        elif self.method == 'val':
             
            w, h = img.size
            new_w = math.ceil(w / 32) * 32
            new_h = math.ceil(h / 32) * 32

            gd_path = os.path.join(self.root, 'gt_points', '{}.npy'.format(name))
            keypoints = np.load(gd_path)

            img = img.resize((640, 384), Image.BICUBIC)
            prev_img = prev_img.resize((640, 384), Image.BICUBIC)
            post_img = post_img.resize((640, 384), Image.BICUBIC)
            return self.w_transform(prev_img),self.w_transform(img),self.w_transform(post_img), len(keypoints), im_path 
        
        elif self.method == 'test':
             
            w, h = img.size
            new_w = math.ceil(w / 32) * 32
            new_h = math.ceil(h / 32) * 32

            gd_path = os.path.join(self.root, 'gt_points', '{}.npy'.format(name))
            keypoints = np.load(gd_path)

            img = img.resize((640, 384), Image.BICUBIC)
            prev_img = prev_img.resize((640, 384), Image.BICUBIC)
            post_img = post_img.resize((640, 384), Image.BICUBIC)
            return self.w_transform(prev_img),self.w_transform(img),self.w_transform(post_img), len(keypoints), im_path 

    def train_transform_density_map(self, img, den_map, label):
        wd, ht = img.size

        if random.random() > 0.88:
            img = img.convert('L').convert('RGB')
        re_size = random.random() * 0.5 + 0.75

        wdd = (int)(wd * re_size)
        htt = (int)(ht * re_size)
        if min(wdd, htt) >= self.c_size:
            wd = wdd
            ht = htt
            img = img.resize((wd, ht))
            den_map = cv2.resize(den_map[:, :], (wd, ht), interpolation=cv2.INTER_CUBIC) / (re_size ** 2)

        
        st_size = min(wd, ht)
        assert st_size >= self.c_size
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        den_map = den_map[i: (i + h), j: (j + w)]
        den_map2 = np.array(den_map)   
        print(den_map.shape)
        print(den_map2.sum())
        den_map = den_map.reshape([h // self.d_ratio, self.d_ratio, w // self.d_ratio, self.d_ratio]).sum(axis=(1, 3))

        if random.random() > 0.5:
            img = F.hflip(img)
            den_map = np.fliplr(den_map)

        return self.w_transform(img), self.s_transform(img), torch.from_numpy(den_map.copy()).float().unsqueeze(0), label
    

    def train_transform_density_map2(self, prev_img, img,  post_img, den_map, label):
        wd, ht = img.size

        den_map2 = np.array(den_map)   
        #print(den_map2.sum())

        if random.random() > 0.88:
            img = img.convert('L').convert('RGB')
            prev_img = prev_img.convert('L').convert('RGB')
            post_img = post_img.convert('L').convert('RGB')
       
        if random.random() > 0.5:
            img = F.hflip(img)
            prev_img = F.hflip(prev_img)
            post_img = F.hflip(post_img)
            den_map = np.fliplr(den_map)


        target = den_map
        target = np.array(target)
        
        c = wd*ht/(64*64)
        target = cv2.resize(target,(64,64),interpolation = cv2.INTER_CUBIC)*c
        # print(target.sum())
        img = img.resize((512,512))
        prev_img = prev_img.resize((512, 512))
        post_img = post_img.resize((512, 512))

        return (
            self.w_transform(prev_img),  # prev_img
            self.w_transform(img),       # current img
            self.w_transform(post_img),  
            self.s_transform(prev_img),   
            self.s_transform(img),       # small img
            self.s_transform(post_img), 
            torch.from_numpy(target.copy()).float().unsqueeze(0),  # den_map
            label
        )

    def train_transform_density_map3(self, prev_img, img, png, post_img, den_map, label):
        wd, ht = img.size

        den_map2 = np.array(den_map)

        # 是否转灰度
        if random.random() > 0.88:
            img = img.convert('L').convert('RGB')
            png = png.convert('L').convert('RGB')
            prev_img = prev_img.convert('L').convert('RGB')
            post_img = post_img.convert('L').convert('RGB')

        # 是否左右翻转
        if random.random() > 0.5:
            img = F.hflip(img)
            png = F.hflip(png)
            prev_img = F.hflip(prev_img)
            post_img = F.hflip(post_img)
            den_map = np.fliplr(den_map)

        # 处理密度图（resize to 64x64）
        target = np.array(den_map)
        # c = wd * ht / (64 * 64)
        # target = cv2.resize(target, (64, 64), interpolation=cv2.INTER_CUBIC) * c

        c = wd * ht / (80 * 48)
        target = cv2.resize(target, (80, 48), interpolation=cv2.INTER_CUBIC) * c


        # 图像 resize 到 512x512
        # img = img.resize((512, 512))
        # prev_img = prev_img.resize((512, 512))
        img = img.resize((640, 384))
        png = png.resize((80, 48))
        prev_img = prev_img.resize((640, 384))
        post_img = post_img.resize((640, 384))
        # 返回处理后的图像对 + 密度图 + 标签
        return (
            self.png_transform(png),
            self.w_transform(prev_img),  # prev_img
            self.w_transform(img),       # current img
            self.w_transform(post_img),  
            self.s_transform(prev_img),   
            self.s_transform(img),       # small img
            self.s_transform(post_img), 
            torch.from_numpy(target.copy()).float().unsqueeze(0),  # den_map
            label
        )

class TwoStreamBatchSampler(data.Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size


        # assert len(self.primary_indices) >= self.primary_batch_size > 0
        # assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)



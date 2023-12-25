from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random
from torchvision import transforms
from PIL import Image

def read_txt(txt_file):
    pd_data = pd.read_csv(txt_file)
    catename2label = pd.read_csv('../data/catename2label_cate8.txt')
    pd_data['label'] = None

    for i in range(len(pd_data)):
        catename = pd_data.loc[i]['catename']
        label = list(catename2label.loc[catename2label['catename'] == catename]['label'])[0]
        pd_data.loc[i,'label'] = label
    return pd_data

class SLC_spe_4D(Dataset):
    def __init__(self, txt_file, spe_dir, spe_transform=None):
        self.data = read_txt(txt_file)
        self.spe_dir = spe_dir
        self.spe_transform = spe_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        slc_spe = np.load(self.spe_dir + self.data.loc[idx]['path'])
        # slc_spe = np.reshape(slc_spe, [32*32, 32, 32])
        catename = self.data.loc[idx]['catename']
        label = self.data.loc[idx]['label']
        #print(slc_spe.shape)(32*32*32*32)
        sample = {'spe': slc_spe,
                  'catename': catename,
                  'label': label,
                  'path': self.data.loc[idx]['path']}
        if self.spe_transform:
            sample['spe'] = self.spe_transform(sample['spe'])

        return sample


class SLC_img(Dataset):
    def __init__(self, txt_file, root_dir, transform = None):
        self.data = read_txt(txt_file)
        self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        slc_data = np.load(self.root_dir + self.data.loc[idx]['path'])
        data = np.log2(np.abs(slc_data) + 1) / 16#(64*64)
        catename = self.data.loc[idx]['catename']
        label = self.data.loc[idx]['label']
        sample = {'data': data,
                  'catename': catename,
                  'label': label,
                  'path': self.data.loc[idx]['path']}

        if self.transform:
            sample['data'] = self.transform(sample['data'])
        #print(sample['data'][0])
        #print(sample['data'][1])
        return sample


class SLC_all_spe(Dataset):
    def __init__(self, txt_file, all_spe_dir, img_dir, all_spe_transform = None,img_transform= None):
        self.data = read_txt(txt_file)
        self.all_spe_dir = all_spe_dir
        self.img_dir = img_dir
        self.all_spe_transform = all_spe_transform
        self.img_transform = img_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = np.load(self.img_dir + self.data.loc[idx]['path'])
        img_data = np.log2(np.abs(img_data) + 1) / 16#(64*64)
        all_spe_data = np.load(self.all_spe_dir+self.data.loc[idx]['path'])

        catename = self.data.loc[idx]['catename']
        label = self.data.loc[idx]['label']
        sample = {'img_data': img_data,
                  'all_spe_data': all_spe_data,
                  'catename': catename,
                  'label': label,
                  'path': self.data.loc[idx]['path']}

        if self.img_transform:
            sample['img_data'] = self.img_transform(sample['img_data'])
        if self.all_spe_transform:
            sample['all_spe_data'] = self.all_spe_transform(sample['all_spe_data'])
        #print(sample['data'][0])
        #print(sample['data'][1])
        return sample
# class SLC_spe_xy(Dataset):
#     def __init__(self, txt_file, spe_dir, spe_transform=None):
#         self.data = read_txt(txt_file)
#         self.spe_dir = spe_dir
#         self.spe_transform = spe_transform
#         print('len(self.data)',len(self.data))
#     def __len__(self):
#         return len(self.data) * 32 * 32
#
#     def __getitem__(self, idx):
#         a = idx // (32*32)
#         b = idx % (32*32)
#
#         slc_spe = np.load(self.spe_dir + self.data.loc[a]['path'])
#         slc_spe = np.reshape(slc_spe, [32*32, 32, 32])
#         catename = self.data.loc[a]['catename']
#         label = self.data.loc[a]['label']
#
#         sample = {'spe': slc_spe[b],
#                   'catename': catename,
#                   'label': label}
#         if self.spe_transform:
#             sample['spe'] = self.spe_transform(sample['spe'])
#         #print(len(sample))
#         return sample

class SLC_spe_xy(Dataset):
    def __init__(self, txt_file, spe_dir, sublook = 32, spe_transform=None):
        self.data = read_txt(txt_file)
        self.spe_dir = spe_dir
        self.spe_transform = spe_transform
        self.sublook = sublook
        #print('len(self.data)',len(self.data))
    def __len__(self):
        return len(self.data) * self.sublook * self.sublook

    def __getitem__(self, idx):
        a = idx // (self.sublook*self.sublook)
        b = idx % (self.sublook*self.sublook)

        slc_spe = np.load(self.spe_dir + self.data.loc[a]['path'])
        slc_spe = np.reshape(slc_spe, [self.sublook*self.sublook, 64-self.sublook, 64-self.sublook])
        catename = self.data.loc[a]['catename']
        label = self.data.loc[a]['label']

        sample = {'spe': slc_spe[b],
                  'catename': catename,
                  'label': label}
        if self.spe_transform:
            sample['spe'] = self.spe_transform(sample['spe'])
        #print('spe',sample['spe'].shape)
        return sample


class SLC_spe(Dataset):
    def __init__(self, txt_file, spe_dir, spe_transform=None):
        self.data = read_txt(txt_file)
        self.spe_dir = spe_dir
        self.spe_transform = spe_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        slc_spe = np.load(self.spe_dir + self.data.loc[idx]['path'])
        slc_spe = np.reshape(slc_spe, [32*32, 32, 32])
        catename = self.data.loc[idx]['catename']
        label = self.data.loc[idx]['label']

        sample = {'spe': slc_spe,
                  'catename': catename,
                  'label': label}
        if self.spe_transform:
            sample['spe'] = self.spe_transform(sample['spe'])
        return sample

class SLC_img_spe4D_single(Dataset):
    def __init__(self, txt_file, img_dir, spe_dir, img_transform=None, spe_transform=None):
        self.data = read_txt(txt_file)
        self.img_dir = img_dir
        self.spe_dir = spe_dir
        self.img_transform = img_transform
        self.spe_transform = spe_transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        slc_img = np.load(self.img_dir + self.data.loc[idx]['path'])
        slc_spe = np.load(self.spe_dir + self.data.loc[idx]['path'])

        slc_img = np.log2(np.abs(slc_img) + 1) / 16

        catename = self.data.loc[idx]['catename']
        label = self.data.loc[idx]['label']
        sample = {'img': slc_img,
                  'spe': slc_spe,
                  'catename': catename,
                  'label': label,
                  'path': self.data.loc[idx]['path']}

        if self.img_transform:
            sample['img'] = self.img_transform(sample['img'])
        if self.spe_transform:
            sample['spe'] = self.spe_transform(sample['spe'])


        return sample

class SLC_img_spe4D(Dataset):
    def __init__(self, txt_file, img_dir, spe_dir_32,spe_dir_48,spe_dir_16, img_transform=None, spe_transform=None):
        self.data = read_txt(txt_file)
        self.img_dir = img_dir
        self.spe_dir_32 = spe_dir_32
        self.spe_dir_48 = spe_dir_48
        self.spe_dir_16 = spe_dir_16
        self.img_transform = img_transform
        self.spe_transform = spe_transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        slc_img = np.load(self.img_dir + self.data.loc[idx]['path'])
        slc_spe_32 = np.load(self.spe_dir_32 + self.data.loc[idx]['path'])
        slc_spe_48 = np.load(self.spe_dir_48 + self.data.loc[idx]['path'])
        slc_spe_16 = np.load(self.spe_dir_16 + self.data.loc[idx]['path'])
        slc_img = np.log2(np.abs(slc_img) + 1) / 16

        catename = self.data.loc[idx]['catename']
        label = self.data.loc[idx]['label']
        sample = {'img': slc_img,
                  'spe_32': slc_spe_32,
                  'spe_48': slc_spe_48,
                  'spe_16': slc_spe_16,
                  'catename': catename,
                  'label': label,
                  'path': self.data.loc[idx]['path']}

        if self.img_transform:
            sample['img'] = self.img_transform(sample['img'])
        if self.spe_transform:
            sample['spe_32'] = self.spe_transform(sample['spe_32'])
            sample['spe_48'] = self.spe_transform(sample['spe_48'])
            sample['spe_16'] = self.spe_transform(sample['spe_16'])


        return sample

# class SLC_img_spe4D_labeled(Dataset):
#     def __init__(self, txt_file, img_dir, spe_dir, aug_img_dir, aug_spe_dir, img_transform=None, spe_transform=None):
#
#         self.data = read_txt(txt_file)
#         self.img_dir = img_dir
#         self.spe_dir = spe_dir
#         self.aug_img_dir = aug_img_dir
#         self.aug_spe_dir = aug_spe_dir
#         self.img_transform = img_transform
#         self.spe_transform = spe_transform
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         if random.random() < 0.5:
#             slc_img = np.load(self.img_dir + self.data.loc[idx]['path'])
#             slc_spe = np.load(self.spe_dir + self.data.loc[idx]['path'])
#         else:
#             slc_img = np.load(self.aug_img_dir + self.data.loc[idx]['path'])
#             slc_spe = np.load(self.aug_spe_dir + self.data.loc[idx]['path'])
#
#         slc_img = np.log2(np.abs(slc_img) + 1) / 16
#
#         catename = self.data.loc[idx]['catename']
#         label = self.data.loc[idx]['label']
#         sample = {'img': slc_img,
#                   'spe': slc_spe,
#                   'catename': catename,
#                   'label': label,
#                   'path': self.data.loc[idx]['path']}
#         if self.img_transform:
#             sample['img'] = self.img_transform(sample['img'])
#         if self.spe_transform:
#             sample['spe'] = self.spe_transform(sample['spe'])
#
#         return sample
#
#
# class SLC_img_spe4D_unlabeled(Dataset):
#     def __init__(self, txt_file, img_dir, spe_dir, aug_img_dir, aug_spe_dir, img_transform=None, spe_transform=None):
#         self.data = read_txt(txt_file)
#         self.img_dir = img_dir
#         self.spe_dir = spe_dir
#         self.aug_img_dir = aug_img_dir
#         self.aug_spe_dir = aug_spe_dir
#         self.img_transform = img_transform
#         self.spe_transform = spe_transform
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         weak_img = np.load(self.img_dir + self.data.loc[idx]['path'])
#         weak_spe = np.load(self.spe_dir + self.data.loc[idx]['path'])
#
#         strong_img = np.load(self.aug_img_dir + self.data.loc[idx]['path'])
#         strong_spe = np.load(self.aug_spe_dir + self.data.loc[idx]['path'])
#
#         weak_img = np.log2(np.abs(weak_img) + 1) / 16
#         strong_img = np.log2(np.abs(strong_img) + 1) / 16
#
#         catename = self.data.loc[idx]['catename']
#         label = self.data.loc[idx]['label']
#         sample = {'weak_img': weak_img,
#                   'weak_spe': weak_spe,
#                   'strong_img': strong_img,
#                   'strong_spe': strong_spe,
#                   'catename': catename,
#                   'label': label,
#                   'path': self.data.loc[idx]['path']}
#
#         if self.img_transform:
#             sample['weak_img'] = self.img_transform(sample['weak_img'])
#             sample['strong_img'] = self.img_transform(sample['strong_img'])
#         if self.spe_transform:
#             sample['weak_spe'] = self.spe_transform(sample['weak_spe'])
#             sample['strong_spe'] = self.spe_transform(sample['strong_spe'])
#         return sample

class SLC_img_spe4D_labeled(Dataset):
    def __init__(self, txt_file, img_dir, spe_dir, num_expand, img_transform=None, spe_transform=None):
        self.data = read_txt(txt_file)
        self.img_dir = img_dir
        self.spe_dir = spe_dir
        self.img_transform = img_transform
        self.spe_transform = spe_transform
        self.transform_labeled = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=64,
                                  padding=int(64 * 0.125))])
        self.index = list(range(len(self.data)))
        self.num_expand = num_expand
        self.labeled_idx = x_u_split(self.index, self.num_expand)

        #print(self.labeled_idx)
        #self.data = self.data.loc[self.labeled_idx]
        #path = {}
        #for i in range(num_expand):

        #print(len(self.data))
        #print(self.data.loc[18]['label'])
    def __len__(self):
        return self.num_expand
        #return len(self.data)

    def __getitem__(self, idx):
        slc_img = np.load(self.img_dir + self.data.loc[self.labeled_idx[idx]]['path'])
        slc_spe = np.load(self.spe_dir + self.data.loc[self.labeled_idx[idx]]['path'])

        slc_img = np.log2(np.abs(slc_img) + 1) / 16

        catename = self.data.loc[self.labeled_idx[idx]]['catename']
        label = self.data.loc[self.labeled_idx[idx]]['label']
        sample = {'img': slc_img,
                  'spe': slc_spe,
                  'catename': catename,
                  'label': label,
                  'path': self.data.loc[self.labeled_idx[idx]]['path']}
        # sample['img'] = Image.fromarray(sample['img'])
        # sample['img'] = self.transform_labeled(sample['img'])
        # sample['img'] = np.array(sample['img'])
        if self.img_transform:
            sample['img'] = self.img_transform(sample['img'])
        if self.spe_transform:
            sample['spe'] = self.spe_transform(sample['spe'])


        return sample

class SLC_img_spe4D_unlabeled(Dataset):
    def __init__(self, txt_file, img_dir, spe_dir, num_expand, img_transform=None, spe_transform=None):
        self.data = read_txt(txt_file)
        self.img_dir = img_dir
        self.spe_dir = spe_dir
        self.img_transform = img_transform
        self.spe_transform = spe_transform
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=64,
                                  padding=int(64 * 0.125))])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=64,
                                  padding=int(64 * 0.125)),
            RandAugmentMC(n=2, m=10)])


        self.index = list(range(len(self.data)))

        self.num_expand = num_expand
        self.unlabeled_idx = x_u_split(self.index, self.num_expand)
        #print(self.labeled_idx.shape)
        #self.data = self.data.loc[self.unlabeled_idx]

    def __len__(self):
        return self.num_expand
        #return len(self.data)

    def __getitem__(self, idx):
        slc_img = np.load(self.img_dir + self.data.loc[self.unlabeled_idx[idx]]['path'])
        slc_spe = np.load(self.spe_dir + self.data.loc[self.unlabeled_idx[idx]]['path'])

        slc_img = np.log2(np.abs(slc_img) + 1) / 16

        catename = self.data.loc[self.unlabeled_idx[idx]]['catename']
        label = self.data.loc[self.unlabeled_idx[idx]]['label']
        sample = {'weak_img': slc_img,
                  'weak_spe': slc_spe,
                  'strong_img':slc_img,
                  'strong_spe':slc_spe,
                  'catename': catename,
                  'label': label,
                  'path': self.data.loc[self.unlabeled_idx[idx]]['path']}
        # sample['weak_img'] = Image.fromarray(sample['weak_img'])
        # sample['weak_img'] = self.weak(sample['weak_img'])
        # sample['weak_img'] = np.array(sample['weak_img'])

        #sample['strong_img'] = Image.fromarray(sample['strong_img']).convert('L')

        #sample['strong_img'] = self.strong(sample['strong_img'])
        #sample['strong_img'] = np.array(sample['strong_img'])
        if self.img_transform:
            sample['weak_img'] = self.img_transform(sample['weak_img'])
            sample['strong_img'] = self.img_transform(sample['strong_img'])
        if self.spe_transform:
            sample['weak_spe'] = self.spe_transform(sample['weak_spe'])
            sample['strong_spe'] = self.spe_transform(sample['strong_spe'])


        return sample

def x_u_split(index,
              num_expand_x):


    exapand_labeled = num_expand_x // len(index)

    labeled_idx = np.hstack(
        [index for _ in range(exapand_labeled)])

    if len(labeled_idx) < num_expand_x:
        diff = num_expand_x - len(labeled_idx)
        labeled_idx = np.hstack(
            (labeled_idx, np.random.choice(labeled_idx, diff)))
    else:
        assert len(labeled_idx) == num_expand_x

    return labeled_idx



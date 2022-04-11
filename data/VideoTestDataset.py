import os.path as osp
import torch
import torch.utils.data as data
import data.util as util
import cv2
import random

class VideoTestDataset(data.Dataset):
    """
    A video test dataset. Support:
    Vid4
    REDS4
    Vimeo90K-Test

    no need to prepare LMDB files
    """

    def __init__(self, opt):
        super(VideoTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_type = opt['data_type']
        self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': [], 'idx': [], 'border': []}
        if self.data_type == 'lmdb':
            raise ValueError('No need to use LMDB during validation/test.')
        #### Generate data info and cache data
        self.imgs_LQ, self.imgs_GT = {}, {}
        if opt['name'].lower() in ['vid4', 'reds4', 'realvsr_test', 'RealVSR_Test']: 
            subfolders_LQ = util.glob_file_list(self.LQ_root)
            subfolders_GT = util.glob_file_list(self.GT_root)
            #print(len(subfolders_LQ), len(subfolders_GT))
            for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
                subfolder_name = osp.basename(subfolder_GT)
                img_paths_LQ = util.glob_file_list(subfolder_LQ)
                img_paths_GT = util.glob_file_list(subfolder_GT)
                #print(img_paths_LQ, sep='\n')
                max_idx = len(img_paths_LQ)
                
                assert max_idx == len(img_paths_GT), 'Different number of images in LQ and GT folders'
                self.data_info['path_LQ'].extend(img_paths_LQ)
                self.data_info['path_GT'].extend(img_paths_GT)
                self.data_info['folder'].extend([subfolder_name] * max_idx)
                for i in range(max_idx):
                    self.data_info['idx'].append('{}/{}'.format(i, max_idx))
                
                border_l = [0] * max_idx
                for i in range(self.half_N_frames):
                    border_l[i] = 1
                    border_l[max_idx - i - 1] = 1
                self.data_info['border'].extend(border_l)
                
                if self.cache_data: # no                
                    self.imgs_LQ[subfolder_name] = util.read_img_seq(img_paths_LQ, color=opt['color'])
                    self.imgs_GT[subfolder_name] = util.read_img_seq(img_paths_GT, color=opt['color'])
               
        elif opt['name'].lower() in ['vimeo90k-test']:
            pass  # TODO
        else:
            raise ValueError(
                'Not support video test dataset. Support Vid4, REDS4 and Vimeo90k-Test.'
            )

    def __getitem__(self, index):
        path_LQ = self.data_info['path_LQ'][index]
        path_GT = self.data_info['path_GT'][index]
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]

        if self.cache_data:
            select_idx = util.index_generation(idx, max_idx, self.opt['N_frames'],
                                               padding=self.opt['padding'])
            imgs_LQ = self.imgs_LQ[folder].index_select(0, torch.LongTensor(select_idx))
            img_GT = self.imgs_GT[folder][idx]
        
        else:
            # my crutch:
            GT_size_tuple = (3, 1024, 512)
            img_LQ = util.read_img(None, path_LQ)[..., ::-1]
            img_LQ_l = cv2.resize(img_LQ, (GT_size_tuple[2] // 2, GT_size_tuple[1] // 2), interpolation=cv2.INTER_LINEAR)#[...,::-1]
            img_GT = util.read_img(None, path_GT)[..., ::-1]
            '''
            GT_size = self.opt['GT_size']
            GT_size_tuple = (3, 1024, 512)
            C, H, W = GT_size_tuple
            rnd_h = random.randint(0, max(0, H - GT_size))
            rnd_w = random.randint(0, max(0, W - GT_size))
            img_LQ = util.read_img(None, path_LQ)[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, ::-1]
            img_LQ_l = cv2.resize(img_LQ, (GT_size // 2, GT_size // 2), interpolation=cv2.INTER_LINEAR)#[...,::-1]
            img_GT = util.read_img(None, path_GT)[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, ::-1]
            '''
            #imgs_LQ = path_LQ
            #img_GT = path_GT

        return {
            'LQ': img_LQ_l,
            'GT': img_GT,
            'folder': folder,
            'idx': self.data_info['idx'][index],
            'border': border
        }

    def __len__(self):
        return len(self.data_info['path_GT'])

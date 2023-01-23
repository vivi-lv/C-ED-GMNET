import torch
import numpy as np
from torch.utils.data import Dataset
import os
from utils.utils import *

def preprocess_test_dataset():
    # scenes_dir = './data/dataset/HDR_Dynamic_Scenes_SIGGRAPH2017/Test/EXTRA'
    scenes_dir = './test_scenes'
    scenes_dir_list = os.listdir(scenes_dir)
    image_list = []
    for scene in range(len(scenes_dir_list)):
        exposure_file_path = os.path.join(scenes_dir, scenes_dir_list[scene], 'exposure.txt')
        ldr_file_path = list_all_files_sorted(os.path.join(scenes_dir, scenes_dir_list[scene]), '.tif')
        #label_path = os.path.join(scenes_dir, scenes_dir_list[scene])
        image_list += [[exposure_file_path, ldr_file_path]]

    sample = []
    for index in range(len(scenes_dir_list)):
        # Read exposure times in one scene
        expoTimes = ReadExpoTimes(image_list[index][0])
        # Read LDR image in one scene
        ldr_images = ReadImages(image_list[index][1])
        # Read HDR label
        #label = ReadLabel(image_list[index][2])
        # ldr images process
        pre_img0 = LDR_to_HDR(ldr_images[0], expoTimes[0], 2.2)
        pre_img1 = LDR_to_HDR(ldr_images[1], expoTimes[1], 2.2)
        pre_img2 = LDR_to_HDR(ldr_images[2], expoTimes[2], 2.2)

        pre_img0 = np.concatenate((ldr_images[0], pre_img0), 2)
        pre_img1 = np.concatenate((ldr_images[1], pre_img1), 2)
        pre_img2 = np.concatenate((ldr_images[2], pre_img2), 2)

        # hdr label process
        #label = range_compressor(label)

        # data argument
        img0 = pre_img0.astype(np.float32).transpose(2, 0, 1)
        img1 = pre_img1.astype(np.float32).transpose(2, 0, 1)
        img2 = pre_img2.astype(np.float32).transpose(2, 0, 1)
        #label = label.astype(np.float32).transpose(2, 0, 1)

        img0 = np.expand_dims(img0, 0)
        img1 = np.expand_dims(img1, 0)
        img2 = np.expand_dims(img2, 0)
        #label = np.expand_dims(label, 0)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        #label = torch.from_numpy(label)

        #sample += [[img0, img1, img2, label]]
        sample += [[img0, img1, img2]]

    return sample, len(scenes_dir_list)
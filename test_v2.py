import torch
from graphs.origin_ahdrnet import AHDR
from test_dataset import preprocess_test_dataset
import numpy as np
import cv2
import sys
from utils.utils import *
import os
from graphs.channel_space_attention_DRDB import CSADRDB
from graphs.cs_unet_DRDB import CSAUNETDRDB
from graphs.merge import MERGE

import argparse
import time


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    return parser.parse_args()

def main():
    args = get_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    start = time.time()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    checkpoint = torch.load('./paper/train_model/merge_drb3_AEDHDR_v2/best_checkpoint_ep5843.pth')
    model = MERGE(6, 5, 64, 32)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    dataset_test, test_num = preprocess_test_dataset()
    result_dir = './paper/result/merge_drb3_AEDHDR/scene'
    result_dir = [f.path for f in os.scandir(result_dir)]
    with torch.no_grad():
        for num in range(test_num):
            ldr0, ldr1, ldr2 = dataset_test[num][0].to(device), dataset_test[num][1].to(device), dataset_test[num][2].to(device)
            pred = model(ldr0, ldr1, ldr2)
            pred_np = pred.detach().cpu().numpy()[0]
            pred_np = pred_np.transpose(1, 2, 0)
            cv2.imwrite(result_dir[num] + '/HDRImg.hdr', pred_np)
            pred_ldr = range_compressor_tensor(pred)
            pred_ldr = torch.clamp(pred_ldr, 0., 1.)
            pred_ldr = pred_ldr.detach().cpu().numpy()[0]
            pred_ldr = pred_ldr.transpose(1, 2, 0)
            cv2.imwrite(result_dir[num] + '/LDRImg.tif', float2int(pred_ldr, np.uint16))
    end = time.time()
    print(end-start)

if __name__ == '__main__':
    main()
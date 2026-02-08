import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
from .data_utils import * 
from .base import BaseDataset
import pandas as pd
from typing import Dict, List


class AllDataset(BaseDataset):
    def __init__(self, image_dir):
        super().__init__()
        self.image_root = image_dir 
        self.size = (1024,1024)


    def __len__(self):
        return len(os.listdir(os.path.join(self.image_root, "target_mask")))

            
    def get_sample(self, idx):
        tar_mask_path = os.path.join(self.image_root, "target_mask")
        data = os.listdir(tar_mask_path)

 

        tar_mask_path = os.path.join(self.image_root, "target_mask", data[idx])
        tar_image_path = tar_mask_path.replace('/target_mask/', '/gt/')
        ref_image_path = tar_mask_path.replace('/target_mask/','/image/')
        ref_mask_path = tar_mask_path.replace('/target_mask/', '/source_mask/')

        ref_image = cv2.imread(ref_image_path)

        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

        tar_image = cv2.imread(tar_image_path)
        tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)


        ref_mask = (cv2.imread(ref_mask_path) > 128).astype(np.uint8)[:,:,0]

        tar_mask = (cv2.imread(tar_mask_path) > 128).astype(np.uint8)[:,:,0]

        

        ref_image = cv2.resize(ref_image.astype(np.uint8), self.size).astype(np.uint8)
        ref_mask = cv2.resize(ref_mask.astype(np.uint8), self.size).astype(np.uint8)
        tar_image = cv2.resize(tar_image.astype(np.uint8), self.size).astype(np.uint8)
        tar_mask = cv2.resize(tar_mask.astype(np.uint8), self.size).astype(np.uint8)


        all_mask = np.logical_or(ref_mask, tar_mask).astype(np.uint8)

        item = self.process_pairs(ref_image, ref_mask, tar_image, tar_mask, all_mask)
        return item




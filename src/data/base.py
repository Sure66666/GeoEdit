import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import cv2
from .data_utils import * 
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from torchvision import transforms, utils
import bezier
import random
import torchvision.transforms as T
import torchvision.utils as vutils
import os
import sys

class BaseDataset(Dataset):
    def __init__(self):
        self.to_tensor = T.ToTensor()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        item = self.get_sample(idx)
        return item
 
    def get_sample(self, idx):
        pass


    def process_pairs(self, ref_image, ref_mask, tar_image, tar_mask, all_mask):


        ref_mask_ori = ref_mask
        if tar_mask.shape != tar_image.shape:
            tar_mask = cv2.resize(tar_mask, (tar_image.shape[1], tar_image.shape[0]))

        ref_box_yyxx = get_bbox_from_mask(ref_mask)
        ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
        masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3) 

        y1,y2,x1,x2 = ref_box_yyxx
        masked_ref_image = masked_ref_image[y1:y2,x1:x2,:] 
        ref_mask = ref_mask[y1:y2,x1:x2] 

        ratio = 1.3
        masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio) 
        masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False) 
        masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), self.size).astype(np.uint8)


        tar_mask_ori = tar_mask
        tar_box_yyxx = get_bbox_from_mask(tar_mask)
        tar_mask_3 = np.stack([tar_mask,tar_mask,tar_mask],-1)
        masked_tar_image = tar_image * tar_mask_3 + np.ones_like(tar_image) * 255 * (1-tar_mask_3) 

        y1,y2,x1,x2 = tar_box_yyxx
        masked_tar_image = masked_tar_image[y1:y2,x1:x2,:] 
        tar_mask = tar_mask[y1:y2,x1:x2] 

        ratio = 1.3
        masked_tar_image, ref_mask = expand_image_mask(masked_tar_image, ref_mask, ratio=ratio) 
        masked_tar_image = pad_to_square(masked_tar_image, pad_value = 255, random = False) 
        masked_tar_image = cv2.resize(masked_tar_image.astype(np.uint8), self.size).astype(np.uint8)

        tar_mask_ori[tar_mask_ori == 1] = 255
        ref_mask_ori[ref_mask_ori == 1] = 127

        all_mask_vis = np.zeros_like(ref_mask_ori, dtype=np.uint8)
        all_mask_vis[ref_mask_ori == 127] = 127
        all_mask_vis[tar_mask_ori == 255] = 255

        all_mask = np.stack([all_mask_vis, all_mask_vis, all_mask_vis], -1)
        tar_mask = np.stack([tar_mask_ori,tar_mask_ori,tar_mask_ori],-1)

        ref_mask = np.stack([ref_mask_ori,ref_mask_ori,ref_mask_ori],-1)


        masked_task_image = ref_image * (1-(tar_mask == 255))
        masked_task_image = pad_to_square(masked_task_image, pad_value = 255, random = False).astype(np.uint8)
        masked_task_image = cv2.resize(masked_task_image.astype(np.uint8), self.size).astype(np.uint8)


        masked_tar_task_image = tar_image * (1-(tar_mask == 255))
        masked_tar_task_image = pad_to_square(masked_tar_task_image, pad_value = 255, random = False).astype(np.uint8)
        masked_tar_task_image = cv2.resize(masked_tar_task_image.astype(np.uint8), self.size).astype(np.uint8)


        masked_all_task_image = ref_image * ((all_mask != 127) & (all_mask != 255))
        masked_all_task_image = pad_to_square(masked_all_task_image, pad_value = 255, random = False).astype(np.uint8)
        masked_all_task_image = cv2.resize(masked_all_task_image.astype(np.uint8), self.size).astype(np.uint8)


        masked_ref_task_image = ref_image * (1-(ref_mask == 255))
        masked_ref_task_image = pad_to_square(masked_ref_task_image, pad_value = 255, random = False).astype(np.uint8)
        masked_ref_task_image = cv2.resize(masked_ref_task_image.astype(np.uint8), self.size).astype(np.uint8)


        tar_image = pad_to_square(tar_image, pad_value = 255, random = False).astype(np.uint8)
        tar_image = cv2.resize(tar_image.astype(np.uint8), self.size).astype(np.uint8)



        tar_mask = pad_to_square(tar_mask, pad_value = 0, random = False).astype(np.uint8)
        tar_mask = cv2.resize(tar_mask.astype(np.uint8), self.size).astype(np.uint8)

        all_mask = pad_to_square(all_mask, pad_value = 0, random = False).astype(np.uint8)
        all_mask = cv2.resize(all_mask.astype(np.uint8), self.size).astype(np.uint8)

        ref_mask = pad_to_square(ref_mask, pad_value = 0, random = False).astype(np.uint8)
        ref_mask = cv2.resize(ref_mask.astype(np.uint8), self.size).astype(np.uint8)
        
        mask_black = np.ones_like(tar_image) * 0

        masked_all_task_image2 = self.to_tensor(np.concatenate([masked_ref_image, masked_all_task_image], axis=1))
        tar_image2 = self.to_tensor(np.concatenate([masked_ref_image, tar_image], axis=1))

        masked_ref_image = self.to_tensor(masked_ref_image)
        masked_ref_task_image = self.to_tensor(masked_ref_task_image)
        all_ref = masked_ref_image

        ref_mask_new = ((ref_mask > 0) & ~(tar_mask > 0)).astype(np.uint8) * 255

        fg_attnmask = np.concatenate([mask_black, tar_mask], axis=1)

        bg_attnmask = np.concatenate([mask_black, ref_mask_new], axis=1)

        all_mask2 = self.to_tensor(np.concatenate([mask_black, all_mask], axis=1))

        
        
        tar_image = self.to_tensor(tar_image)
        all_mask = self.to_tensor(all_mask)

        item = dict(
                ref=all_ref,   
                src=masked_all_task_image2, 
                result=tar_image2,
                mask=all_mask2,  
                fg_attnmask=fg_attnmask,
                bg_attnmask=bg_attnmask
                ) 
        
        return item


   
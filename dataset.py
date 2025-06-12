import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import re

class RoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        def get_numeric_part(filename):
            numeric_parts = re.findall(r'\d+', os.path.splitext(filename)[0])
            if numeric_parts:
                return int(numeric_parts[-1])
            raise ValueError(f"Filename '{filename}' does not contain a numeric part for sorting.")
            
        self.image_filenames = sorted(os.listdir(self.image_dir), 
                                      key=get_numeric_part)

    def __getitem__(self, idx):
        img_filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, img_filename)

        numeric_id_match = re.search(r'\d+', os.path.splitext(img_filename)[0])
        if not numeric_id_match:
            raise ValueError(f"Could not extract numeric ID from image filename: {img_filename}")
        
        numeric_id = numeric_id_match.group(0)

        mask_filename = f"{numeric_id}.bmp" 
        mask_path = os.path.join(self.mask_dir, mask_filename)

        # --- Kiểm tra sự tồn tại của file (Giữ lại để gỡ lỗi) ---
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path} (corresponding to image: {img_filename})")
        # -----------------------------------------------------------

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        
        y1 = mask.type(torch.BoolTensor) 
        y2 = torch.bitwise_not(y1) 
        mask = torch.cat([y2, y1], dim=0) 
        
        return image, mask
            
    def __len__(self):
        return len(self.image_filenames)

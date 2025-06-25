import os
import cv2

import torch
from torch.utils.data import Dataset

class SRCNNTrainDataset(Dataset):
    def __init__(self, image_dir):
        self.truth_path = image_dir + '/' + 'train/' + 'high_res/' 
        self.truth_images_names = [os.path.join(self.truth_path, image_file_names) for image_file_names in os.listdir(self.truth_path)]
        self.lr_path = image_dir + '/' + 'train/' + 'low_res/'
        self.lr_images_names = [os.path.join(self.lr_path, image_file_names) for image_file_names in os.listdir(self.lr_path)]
        
    def __getitem__(self, index):
        truth_image = cv2.imread(self.truth_images_names[index]) #numpy ndarray
        lr_image = cv2.imread(self.lr_images_names[index]) #numpy ndarray

        ycbcr_truth = cv2.cvtColor(truth_image, cv2.COLOR_BGR2YCR_CB)
        y_truth, cr_truth, cb_truth = cv2.split(ycbcr_truth)
        ycbcr_lr = cv2.cvtColor(lr_image, cv2.COLOR_BGR2YCR_CB)
        y_lr, cr_lr, cb_lr = cv2.split(ycbcr_lr)

        y_truth = torch.from_numpy(y_truth.astype('float32'))
        y_truth -= torch.min(y_truth)
        y_truth /= torch.max(y_truth)

        y_lr = torch.from_numpy(y_lr.astype('float32'))
        y_lr -= torch.min(y_lr)
        y_lr /= torch.max(y_lr)

        y_truth = torch.unsqueeze(y_truth, 0)
        y_lr = torch.unsqueeze(y_lr, 0)

        return {"lr" : y_lr, "truth" : y_truth}
    
    def __len__(self):
        return len(self.truth_images_names)

class SRCNNValDataset(Dataset):
    def __init__(self, image_dir):
        self.truth_path = image_dir + '/' + 'val/' + 'high_res/' 
        self.truth_images_names = [os.path.join(self.truth_path, image_file_names) for image_file_names in os.listdir(self.truth_path)]
        self.lr_path = image_dir + '/' + 'val/' + 'low_res/'
        self.lr_images_names = [os.path.join(self.lr_path, image_file_names) for image_file_names in os.listdir(self.lr_path)]
        
    def __getitem__(self, index):
        truth_image = cv2.imread(self.truth_images_names[index]) #numpy ndarray
        lr_image = cv2.imread(self.lr_images_names[index]) #numpy ndarray

        ycbcr_truth = cv2.cvtColor(truth_image, cv2.COLOR_BGR2YCR_CB)
        y_truth, cr_truth, cb_truth = cv2.split(ycbcr_truth)
        ycbcr_lr = cv2.cvtColor(lr_image, cv2.COLOR_BGR2YCR_CB)
        y_lr, cr_lr, cb_lr = cv2.split(ycbcr_lr)

        y_truth = torch.from_numpy(y_truth.astype('float32'))
        y_truth -= torch.min(y_truth)
        y_truth /= torch.max(y_truth)
        y_lr = torch.from_numpy(y_lr.astype('float32'))
        y_lr -= torch.min(y_lr)
        y_lr /= torch.max(y_lr)
        y_truth = torch.unsqueeze(y_truth, 0)
        y_lr = torch.unsqueeze(y_lr, 0)

        return {"lr" : y_lr, "truth" : y_truth}
    
    def __len__(self):
        return len(self.truth_images_names)
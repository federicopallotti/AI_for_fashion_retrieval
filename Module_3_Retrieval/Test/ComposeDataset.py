import os

import torch
from torch.utils.data import  Dataset
from torchvision.io import read_image
import pandas as pd
import random

#COMMENTED LINES ARE PREVIOUS VERSION OF THE DATALOADER

# dataset_path = '../DatasetFolder/DatasetCV'
#
# class CustomImageDataset(Dataset):
#     def __init__(self, annotations_file, root_dir, transform=None, target_transform=None):
#         self.img_labels = pd.read_csv(annotations_file, sep=" ")
#         self.root_dir = root_dir
#         self.transform = transform
#         self.target_transform = target_transform
#
#     def __len__(self):
#         return len(self.img_labels)
#
#     def __getitem__(self, idx):
#         img_path = os.path.join(self.root_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label
#
# class FeedCoupleDataset(Dataset):
#     printflag = False
#     def __init__(self, annotations_file, root_dir, transform_items=None,
#                  transform_parsed=None):
#
#         self.img_labels = pd.read_csv(annotations_file, sep=' ', header=None)
#         self.root_dir = root_dir
#         self.transform_items = transform_items
#         self.transform_parsed = transform_parsed
#
#     def set_flag(self, flag):
#         self.printflag = flag
#
#     def __len__(self):
#         return len(self.img_labels)
#
#     def __getitem__(self, idx):
#         #record1 = random.randint(0, self.img_labels.shape[0] - 1)
#         image0_image1 = self.img_labels.iloc[idx, :]
#         if self.printflag:
#             print('Immagine0 : ', image0_image1[0])
#             print('Immagine1 : ', image0_image1[1])
#
#         img0 = read_image(os.path.join(self.root_dir, str(image0_image1[0].split('/')[0]), str(image0_image1[0].split('/')[1]))).type(torch.float32)
#         img1 = read_image(os.path.join(self.root_dir, str(image0_image1[1].split('/')[0]), str(image0_image1[1].split('/')[1]))).type(torch.float32)
#
#         if self.transform_parsed and self.transform_items is not None:
#             img0 = self.transform_parsed(img0)
#             img1 = self.transform_items(img1)
#
#         return img0, img1
#
#
# class FeedDatasetTesting(Dataset):
#     printflag = False
#     def __init__(self, annotations_file, root_dir, image_path : str, transform_items=None):
#
#         self.img_labels = pd.read_csv(annotations_file, sep=' ', header=None)
#         self.root_dir = root_dir
#         self.transform_items = transform_items
#         self.image_path = image_path
#         series = self.img_labels.iloc[:, 0]
#         valid = series.str.contains(self.image_path, regex=False)
#         self.img_labels = self.img_labels[valid]
#
#
#     def set_flag(self, flag):
#         self.printflag = flag
#
#     def __len__(self):
#         return len(self.img_labels)
#
#     def __getitem__(self, idx):
#
#         image0_image1 = self.img_labels.iloc[idx, :]
#
#         if self.printflag:
#             print('Immagine0 : ', image0_image1[0])
#             print('Immagine1 : ', image0_image1[1])
#
#         img0 = read_image(os.path.join(self.root_dir, str(image0_image1[0].split('/')[0]), str(image0_image1[0].split('/')[1]))).type(torch.float32)
#         img1 = read_image(os.path.join(self.root_dir, str(image0_image1[1].split('/')[0]), str(image0_image1[1].split('/')[1]))).type(torch.float32)
#
#         if self.transform_items is not None:
#             img1 = self.transform_items(img1)
#
#         return img1



class FeedDatasetTestingRecall(Dataset):
    printflag = False
    def __init__(self, annotations_file, root_dir, category : str, filtering : bool = False, transform_items=None):

        self.img_labels = pd.read_csv(annotations_file, sep=' ', header=None)
        self.annotations_file = annotations_file
        self.root_dir = root_dir
        self.category= category
        self.transform_item = transform_items

        if filtering:
            series = self.img_labels.iloc[:, 0]
            valid = series.str.contains(self.category, regex=False)
            self.img_labels = self.img_labels[valid]


    def set_flag(self, flag):
        self.printflag = flag

    def __len__(self):
        return len(self.img_labels)

    @staticmethod
    def select_random(category, annotations_file, filtering : bool = False):
        '''

        :return: la coppia immagine 0 immagine 1
        '''

        img_labels = pd.read_csv(annotations_file, sep=' ', header=None)

        if filtering:
            series = img_labels.iloc[:, 0]
            valid = series.str.contains(category, regex=False)
            img_labels = img_labels[valid]

        rand_record = random.randint(0, img_labels.shape[0] - 1)
        record = img_labels.iloc[rand_record, :]

        return record[0], record[1]

    @staticmethod
    def get(annotations_file, idx):

        img_labels = pd.read_csv(annotations_file, sep=' ', header=None)


        record = img_labels.iloc[idx, :]

        return record[0], record[1]

    def __getitem__(self, idx):

        image0_image1 = self.img_labels.iloc[idx, :]

        if self.printflag:
            print('Immagine0 : ', image0_image1[0])
            print('Immagine1 : ', image0_image1[1])

        img0 = read_image(os.path.join(self.root_dir, str(image0_image1[0].split('/')[0]), str(image0_image1[0].split('/')[1]))).type(torch.float32)
        img1 = read_image(os.path.join(self.root_dir, str(image0_image1[1].split('/')[0]), str(image0_image1[1].split('/')[1]))).type(torch.float32)

        if self.transform_item is not None:
            img1 = self.transform_item(img1)

        return img1, str(image0_image1[0]), str(image0_image1[1])

def import_classes(input_file : str, classes : dict):
    i = 0
    with open(input_file, "r") as file:
        for line in file:
            classes[i] = str(line.split('\n')[0])
            i += 1
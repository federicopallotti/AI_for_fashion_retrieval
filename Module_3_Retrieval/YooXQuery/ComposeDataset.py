import os
import torch
from torch.utils.data import  Dataset
from torchvision.io import read_image
import pandas as pd
import random


# COMMENTED CODE IS FROM A PREVIOUS VERSION

# class FeedDatasetTestingRecall(Dataset):
#     printflag = False
#     def __init__(self, annotations_file, root_dir, category : str, transform_items=None):
#
#         self.img_labels = pd.read_csv(annotations_file, sep=' ', header=None)
#         self.annotations_file = annotations_file
#         self.root_dir = root_dir
#         self.category= category
#         self.transform_item = transform_items
#
#         series = self.img_labels.iloc[:, 0]
#         valid = series.str.contains(self.category, regex=False)
#         self.img_labels = self.img_labels[valid]
#
#
#     def set_flag(self, flag):
#         self.printflag = flag
#
#     def __len__(self):
#         return len(self.img_labels)
#
#     @staticmethod
#     def select_random(category, annotations_file):
#         '''
#
#         :return: la coppia immagine 0 immagine 1
#         '''
#
#         img_labels = pd.read_csv(annotations_file, sep=' ', header=None)
#         series = img_labels.iloc[:, 0]
#         valid = series.str.contains(category, regex=False)
#         img_labels = img_labels[valid]
#
#         rand_record = random.randint(0, img_labels.shape[0] - 1)
#         record = img_labels.iloc[rand_record, :]
#
#         return record[0], record[1]
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
#         if self.transform_item is not None:
#             img1 = self.transform_item(img1)
#
#         return img1, str(image0_image1[0]), str(image0_image1[1])

class FeedImagesToInput(Dataset):
    printflag = False
    def __init__(self, annotations_file, root_dir, category : str, transform_items=None, filtering : bool = False):

        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.annotations_file = annotations_file
        self.root_dir = root_dir
        self.category= category
        self.transform_item = transform_items

        if filtering:
            series = self.img_labels.iloc[:, 0]
            valid = series.str.contains(self.category, regex=True)
            self.img_labels = self.img_labels[valid]
        print('dimensione dei confronti : ',  self.img_labels.shape[0])
    def set_flag(self, flag):
        self.printflag = flag

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        image = self.img_labels.iloc[idx, 0]

        if self.printflag:
            print('Immagine0 : ', image)

        splitted = str(image).split(os.sep)

        img = read_image(os.path.join(self.root_dir, splitted[0], splitted[1], splitted[2])).type(torch.float32)

        if self.transform_item is not None:
            img = self.transform_item(img)

        return img

def import_classes(input_file : str, classes : dict):
    i = 0
    with open(input_file, "r") as file:
        for line in file:
            classes[i] = str(line.split('\n')[0])
            i += 1
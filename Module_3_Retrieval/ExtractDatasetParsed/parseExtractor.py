from extractorUtils import get_parsed,parse_item,load_palette, get_part, get_newpath
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torchvision.transforms as transforms
import torchvision
import os
import torch
import sys

class Extractor():
    def __init__(self, hyper):

        bools = {'True' : True, 'False' : False}

        self.verbose = bools[(hyper['verbose'])]

        self.root_dir = hyper['root_dir']

        self.category = hyper['category']

        self.dataset = hyper['dataset']

        csvs = {'test' : 'test_pairs_paired.txt', 'train' : 'train_pairs.txt'}

        self.csv_path = os.path.join(self.root_dir, self.category, csvs.get(self.dataset, 'train_pairs.txt'))

        self.data = pd.read_csv(self.csv_path, header=None, sep=' ')

        self.bb = f'{self.category}/images/'

        self.base_path = os.path.join(self.root_dir, self.category, 'images')

        self.palette_file = os.path.join(self.root_dir, 'palette.txt')

        self.csvpath = os.path.join(self.root_dir, 'yu-vton.csv')

        self.palette = load_palette(self.palette_file)

        self.preprocess = torchvision.transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((650, 522)),
            transforms.CenterCrop(224),
        ])

    def perform(self):
        for i in range(self.data.shape[0]):
            img0 = self.data.iloc[i, 0]
            img1 = self.data.iloc[i, 1]
            part = get_part(self.base_path)
            parsed_img, target_img = get_parsed(img1, root_dir=self.root_dir, csvpath=self.csvpath, base_path=self.bb,
                                                verbose=self.verbose)
            parsed_img = np.delete(parsed_img, 3, 2)  # deleting the fourth useless channel
            parsed_img = parsed_img * 255

            parsed_img = parsed_img.astype(np.int64)
            target_img = target_img.astype(np.int64)

            item, mask = parse_item(label=part, parsed=parsed_img, img=target_img.copy(), palette=self.palette)

            item = item.numpy().astype(np.uint8)

            mask = torch.where(mask == 0, mask, 1).numpy().astype(np.uint8)

            contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            if len(contours) == 0:
                print('No ', part, ' found')
                # sys.exit(2)
                continue

            areas = []
            for j in range(len(contours)):
                rect = cv2.boundingRect(contours[j])
                area = rect[2] * rect[3]
                areas.append([area])

            areas = np.array(areas)
            index = np.argmax(areas)

            rect = cv2.boundingRect(contours[index])

            x, y, w, h = rect

            # cropping the image ...
            cropped = item[y:y + h + 0, x - 0: x + w + 0]

            img = np.zeros((1024, 768, 3), dtype=np.uint8)
            img += 255  # color values restoring

            ih = (1024 // 2)
            iw = (768 // 2)
            ih = ih - (cropped.shape[0] // 2)
            iw = iw - (cropped.shape[1] // 2)


            img[ih: ih + cropped.shape[0], iw: iw + cropped.shape[1], :] = cropped[:, :, :]



            img = self.preprocess(img)
            img *= 255  # color map restoring after floating point computation
            img = img.type(torch.uint8)

            if self.verbose:
                plt.imshow(img.permute(1, 2, 0).numpy())
                plt.show()
                print(img.shape)

            new_path = get_newpath(self.base_path, img0)

            if self.verbose:
                print(new_path)

            plt.imsave(str(new_path), img.permute(1, 2, 0).numpy())







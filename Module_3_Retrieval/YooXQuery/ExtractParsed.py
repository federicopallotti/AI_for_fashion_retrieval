from UtilsExtractParsed import make_csv, get_parsed, parse_item
import json
import cv2
import torchvision
from torchvision import transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

annotation_file = make_csv()

p_bermudas = torchvision.transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(500),
    transforms.CenterCrop(224)
])

p_upper = torchvision.transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(800),
    transforms.CenterCrop(224)
])

preprocess = {'bermudas' : p_bermudas, 'upper' : p_upper}

def make(Fields, category, part):
    data = pd.read_csv(annotation_file, header=None, sep=',')
    f = open('palette.json', 'r')
    palette = json.load(f)

    for i in range(data.shape[0]):
        target_img, parsed_img = get_parsed(data.iloc[i, 0], annotation_file)

        parsed_img = np.delete(parsed_img, 3, 2)  # deleting useless fourth channel
        # target_img = np.delete(target_img, 3, 2) # in some input images (especially when in png format) could be required to delete an additional fourth channel

        parsed_img = parsed_img * 255
        # target_img = target_img * 255

        parsed_img = parsed_img.astype(np.int64)

        target_img = target_img.astype(np.int64)


        item, mask = parse_item(label=part, parsed=parsed_img, img=target_img.copy(), palette=palette)

        item = item.numpy().astype(np.uint8)
        mask = torch.where(mask == 0, mask, 1).numpy().astype(np.uint8)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        if len(contours) == 0:
            print('No ', part, ' found')
            # sys.exit(2)
            continue

        index = -1
        areas = []
        for j in range(len(contours)):
            rect = cv2.boundingRect(contours[j])
            area = rect[2] * rect[3]
            areas.append([area])

        areas = np.array(areas)
        index = np.argmax(areas)

        rect = cv2.boundingRect(contours[index])

        x, y, w, h = rect

        rect2 = cv2.rectangle(item.copy(), (x, y), (x + w, y + h), (200, 0, 0), 2)

        plt.imshow(rect2)
        plt.show()

        cropped = item[y:y + h + 0, x - 0: x + w + 0]

        H = target_img.shape[0]
        W = target_img.shape[1]

        img = np.zeros((H, W, 3), dtype=np.uint8)
        img += 255

        ih = (H // 2)
        iw = (W // 2)
        ih = ih - (cropped.shape[0] // 2)
        iw = iw - (cropped.shape[1] // 2)

        ones = np.ones((cropped.shape[0], cropped.shape[1], cropped.shape[2]), dtype=np.uint8)
        img[ih: ih + cropped.shape[0], iw: iw + cropped.shape[1], :] = cropped[:, :, :]

        img = preprocess[Fields](img)
        # img1_im = pp(img1_im)
        img *= 255
        img = img.type(torch.uint8)
        img = img.permute(1, 2, 0).numpy()

        plt.imshow(img)
        plt.show()

        folder_to_save = f'inputs/{category}'
        name = data.iloc[i, 0].split(os.sep)[1]
        name = name.split('.')[0]
        name += '.jpg'
        plt.imsave(os.path.join(folder_to_save, name), img)


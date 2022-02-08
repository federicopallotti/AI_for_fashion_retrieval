import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy
import json

def make_csv():
    files_images = []
    with os.scandir("images") as it:
        for entry in it:
            if entry.is_file():
                #print(entry.name)
                files_images.append(entry.name)

    files_parsed = []
    with os.scandir("parsed") as it:
        for entry in it:
            if entry.is_file():
                #print(entry.name)
                files_parsed.append(entry.name)

    filename = "temporal_csv.txt"

    with open(filename, "w") as f:
        for i in range(len(files_images)):
            f.write(os.path.join("images", str(files_images[i])))
            f.write(',')
            f.write(os.path.join("parsed", str(files_parsed[i])))
            f.write('\n')

    return filename

def make_input_csv(category : str):
    files_images = []
    with os.scandir(f"inputs/{category}") as it:
        for entry in it:
            if entry.is_file():
                files_images.append(entry.name)

    filename = "inputs_csv.txt"

    with open(filename, "w") as f:
        for i in range(len(files_images)):
            name = str(files_images[i]).split('.')[0]
            name = name + '.jpg'
            f.write(os.path.join(f"inputs/{category}", name))
            f.write(',\n')
    return filename


def get_parsed(record : str, csvpath : str) -> torch.Tensor:

    data = pd.read_csv(csvpath, header=None, sep=',')
    imgs = data[data[0] == record]
    img0 = plt.imread(imgs[0].item())
    img1 = plt.imread(imgs[1].item())
    return img0, img1


def parse_item(label : str, parsed : numpy.ndarray, img : numpy.ndarray, palette : dict):
    """
    :param label : class to get parsed
    :param parsed: Tensor dim = (H, W, C):
    :param img: original image to extract the segmentation
    :param palette: color palette of the parser
    :return img, mask_item : tuple containing the image and the parsed mask
    """

    parsed = torch.from_numpy(parsed)
    img = torch.from_numpy(img)
    t = torch.tensor(palette[label], dtype=torch.int64)
    t = torch.reshape(t, (1, 1, 3))

    mask_item = parsed == t
    mask_item = torch.prod(mask_item, dim = 2)
    img[:, :, 0] = torch.where(mask_item == 1, img[:, :, 0], 255)
    img[:, :, 1] = torch.where(mask_item == 1, img[:, :, 1], 255)
    img[:, :, 2] = torch.where(mask_item == 1, img[:, :, 2], 255)

    return img, mask_item

def load_palette(palette_file : str,dump = False):
    '''
    :param palette_file: filename of the palette
    :param dump: boolean to dump the palette dictionary into a json file
    :return: palette dictionary
    '''

    palette = {}
    with open(palette_file, 'r') as file:
        for line in file:
            k = line.split(':')[0].split('\'')[0]
            if k == '{' or k == '':
                k = line.split(':')[0].split('\'')[1]
            #splitted = line.split(':')[1].split(',')
            v1 = int(line.split(':')[1].split(',')[0].split('[')[1])
            v2 = int(line.split(':')[1].split(',')[1])
            v3 = int(line.split(':')[1].split(',')[2].split(']')[0])
            v = [v1, v2, v3]
            palette[k] = v

    if dump:
        with open('palette.json', 'w', encoding='utf-8') as f:
            json.dump(palette, f, ensure_ascii=False, indent=4)

    return palette




import matplotlib.pyplot as plt
import numpy
import pandas as pd
import os
import json
import torch


def get_parsed(record : str, root_dir : str, csvpath : str, base_path : str, verbose) -> torch.Tensor:
    '''
    :param verbose: if verbose or not
    :param record: the item record in the csv
    :param root_dir: root directory of the dataset
    :param csvpath: the path of the principal csv "yu-vton.csv"
    :param base_path: the default folder of the images "lower_body/images/"
    :return: (parsed item, entire mannequin image)
    '''
    record = base_path + record

    data = pd.read_csv(csvpath, sep=',')

    rec = data[data['garment'] == record]


    target = rec['target'].item()
    parsed = rec['parsing'].item()

    sep = parsed.split('/')
    parsed = os.path.join(root_dir, sep[0])
    for i in range(1, len(sep)):
        parsed = os.path.join(parsed, sep[i])


    sep = target.split('/')
    target = os.path.join(root_dir, sep[0])
    for i in range(1, len(sep)):
        target = os.path.join(target, sep[i])

    if verbose:
        print('path of the parsed img : ', parsed)
        print('path of the target image :', target)

    parsed_img = plt.imread(parsed)
    target_img = plt.imread(target)

    return parsed_img, target_img

def parse_item(label : str, parsed : numpy.ndarray, img : numpy.ndarray, palette : dict):
    """
    :param label : class to get parsed
    :param parsed: Tensor dim = (H, W, C):
    :param img: original image where extract the segmentation
    :param palette: color palette of the parser
    :return (img, mask_item) : a pair containing the image and the mask of the item
    """

    parsed = torch.from_numpy(parsed)
    img = torch.from_numpy(img)
    t = torch.tensor(palette[label], dtype=torch.int64)
    t = torch.reshape(t, (1, 1, 3))

    mask_item = parsed == t
    mask_item = torch.prod(mask_item, dim = 2)
    img[: , :, 0] = torch.where(mask_item == 1, img[: , :, 0], 255)
    img[:, :, 1] = torch.where(mask_item == 1, img[:, :, 1], 255)
    img[:, :, 2] = torch.where(mask_item == 1, img[:, :, 2], 255)

    return img, mask_item

def load_palette(palette_file : str,dump = False):
    '''
    :param palette_file: the palette file
    :param dump: boolean value to dump the palette in a json file or not
    :return: the palette dictionary
    '''

    palette = {}
    with open(palette_file, 'r') as file:
        for line in file:
            k = line.split(':')[0].split('\'')[0]
            if k == '{' or k == '':
                k = line.split(':')[0].split('\'')[1]
            v1 = int(line.split(':')[1].split(',')[0].split('[')[1])
            v2 = int(line.split(':')[1].split(',')[1])
            v3 = int(line.split(':')[1].split(',')[2].split(']')[0])
            v = [v1, v2, v3]
            palette[k] = v

    if dump:
        with open('palette.json', 'w', encoding='utf-8') as f:
            json.dump(palette, f, ensure_ascii=False, indent=4)

    return palette


def get_part(base_path):
    if 'lower_body' in base_path:
        return 'Pants'
    elif 'upper_body' in base_path:
        return 'Upper-clothes'

def get_newpath(base : str, path_img0 : str):
    arr = path_img0.split('/')
    path = os.path.join(base, arr[0])
    path = os.path.join(path, arr[1])
    return path

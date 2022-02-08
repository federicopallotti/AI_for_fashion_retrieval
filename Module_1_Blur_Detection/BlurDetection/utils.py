import numpy as np
import sys
from tqdm import tqdm
import cv2

def FrameExtractor(path : str, number : int, sec_skip : int, sec_start_point : int) -> np.ndarray:
    '''
    Function that extracts 'number' images from the path video and returns a batch of frames.
    :param path: video path
    :param number: number of frames to extract
    :param sec_skip: number os seconds to skip between frames
    :param sec_start_point: second to start to sample
    :return: batch of extracted frames of dimensions  (number, 720, 1280, 3)
    '''

    videocap = cv2.VideoCapture(path)
    videocap.set(0, sec_start_point * 1000)

    images = np.zeros((1, 540, 960, 3))
    count = 1
    success = True
    for i in tqdm(range(number)):
        success, image = videocap.read()
        videocap.set(0, (sec_start_point * 1000) + sec_skip * count * 1000)
        if not success:
            print("Insucces Video Read...")
            sys.exit(-1)
        images = np.concatenate((images, image[np.newaxis]), axis=0)
        count += 1

    videocap.release()

    images = images[1::]

    return images

import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import torchvision
from utils import FrameExtractor

def extractor(path, mode, kernel_n, parameters_mode1, parameters_mode2, threshold, format):

    if mode == 1:
        images = torchvision.io.read_video(path, start_pts=parameters_mode1['start_pts'], end_pts=parameters_mode1['end_pts'], pts_unit='sec')
        images = images[0]
        n = parameters_mode1['skip']
        images = images[::n]
        images = images.numpy()
    elif mode == 2:
        number_of_images = parameters_mode2['number_of_images']
        seconds_to_skip = parameters_mode2['seconds_to_skip']
        sec_start_point = parameters_mode2['sec_start_point']
        images = FrameExtractor(path, number_of_images, seconds_to_skip, sec_start_point)

    print(images.shape)


    images = cv2.convertScaleAbs(images)
    for i in range(images.shape[0]):
        # print(images[i].shape)
        images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        cv2.imshow("Image" + str(i + 1), images[i])
        if cv2.waitKey(0) == ord('q'):
            break
        cv2.destroyAllWindows()
    cv2.destroyAllWindows()

    final_images = images.copy()
    batch = images
    images = np.ones((1, batch.shape[1], batch.shape[2]))
    for img in batch:
        # print(img.shape)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = gray.reshape((1, batch.shape[1], batch.shape[2]))

        images = np.concatenate((images, gray), axis=(0))

    images = images[1:]

    images = cv2.convertScaleAbs(images)
    for i in range(images.shape[0]):
        cv2.imshow("Image", images[i])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    images = images[:, np.newaxis]

    def convolution(input, kernel):
        if (torch.cuda.is_available()):
            input = input.cuda()
            kernel = kernel.cuda()
            print('CUDA activated! ' + 'input : ' + str(input.device) + '  kernel  ' + str(kernel.device))

        n, iC, H, W = input.shape
        kH = kernel.shape[2]
        kW = kernel.shape[3]
        oC = kernel.shape[0]

        out = torch.zeros((n, oC, (H - kH + 1), (W - kW + 1)))
        if (torch.cuda.is_available()):
            out = out.cuda()
            print('CUDA activated out! ' + '  out : ' + str(out.device))
        for r in range(0, H - kH + 1):
            for c in range(0, W - kW + 1):
                out[:, :, r, c] = (input[:, :, r:r + kH, c:c + kW].reshape((n, 1, iC, kH, kW)) * kernel.reshape(
                    (1, oC, iC, kH, kW))).sum((2, 3, 4))
        return out


    if kernel_n == 1:
        kernel = np.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]], dtype=np.float32)
    elif kernel_n == 2:
        kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]], dtype=np.float32)
    else:
        print('error selecting kernel')
        sys.exit(-1)

    print(kernel)

    k = kernel + np.zeros((3, 3))
    k = k[np.newaxis, np.newaxis]

    o = convolution(torch.from_numpy(images), torch.from_numpy(k))

    if torch.cuda.is_available():
        print('Cuda available ' + str(o.device))
        o = o.cpu().numpy()
    else:
        o = o.numpy()

    o = np.squeeze(o)
    o = cv2.convertScaleAbs(o)
    for i in range(o.shape[0]):
        cv2.imshow("Image" + str(i), o[i])

        if cv2.waitKey(0) == ord('q'):
            break
        cv2.destroyAllWindows()
    cv2.destroyAllWindows()

    thresh = threshold

    vars = np.var(o, axis=(1, 2))

    mask = vars >= thresh

    m = o.copy()

    t = ['Blurry', 'Not Blurry']
    for i in range(m.shape[0]):
        if vars[i] < thresh:
            text = t[0]
        else:
            text = t[1]
        r = cv2.putText(m[i], "{}: {:.2f}".format(text, vars[i]), (0, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3,
                        bottomLeftOrigin=False, )
        cv2.imshow("Image" + str(i + 1), r)
        if cv2.waitKey(0) == ord('q'):
            break
        cv2.destroyAllWindows()
    cv2.destroyAllWindows()

    final_images = final_images[mask]
    print(final_images.shape)

    for i in range(final_images.shape[0]):
        # final_images[i] = cv2.cvtColor(final_images[i], cv2.COLOR_BGR2RGB)
        cv2.imshow("Image" + str(i), final_images[i])
        if cv2.waitKey(0) == ord('q'):
            break
        cv2.destroyAllWindows()
    cv2.destroyAllWindows()

    path_folder = 'Images_inputParser'
    name = 'input'
    for i in range(final_images.shape[0]):
        path = os.path.join(path_folder, name + str(i) + '.' + format)
        plt.imsave(path, final_images[i])



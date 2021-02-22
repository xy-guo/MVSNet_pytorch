import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt

import sys
sys.path.append("../datasets")
from data_io import read_pfm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('depth_path')
    args = parser.parse_args()
    depth_path = args.depth_path
    if depth_path.endswith('npy'):
        depth_image = np.load(depth_path)
        depth_image = np.squeeze(depth_image)
        print('value range: ', depth_image.min(), depth_image.max())
        plt.imshow(depth_image, 'rainbow')
        plt.show()
    elif depth_path.endswith('pfm'):
        depth_image, _ = read_pfm(depth_path)
        print('depth shape: {}'.format(depth_image.shape))
        ma = np.ma.masked_equal(depth_image, 0.0, copy=False)
        print('value range: ', ma.min(), ma.max())
        plt.imshow(depth_image, 'rainbow')
        plt.show()
    else:
        depth_image = cv2.imread(depth_path)
        ma = np.ma.masked_equal(depth_image, 0.0, copy=False)
        print('value range: ', ma.min(), ma.max())
        plt.imshow(depth_image)
        plt.show()

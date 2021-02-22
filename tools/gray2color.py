import numpy as np
import cv2
import argparse

import sys
sys.path.append("../datasets")
from data_io import read_pfm#, write_depth_img


def write_depth_img(filename, depth_image):
    # Mask the array where equal to a given value
    ma = np.ma.masked_equal(depth_image, 0.0, copy=False)
    d_min = ma.min()
    d_max = ma.max()
    depth_n = 255.0 * (depth_image - d_min) / (d_max - d_min) # depth map normalize
    depth_n = depth_n.astype(np.uint8)
    out_depth_image = cv2.applyColorMap(depth_n, cv2.COLORMAP_JET) # applyColorMap
    cv2.imwrite(filename, out_depth_image)    


if __name__ == "__main__":
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("depth_path")
    args = parser.parse_args()
    depth_path = args.depth_path
    
    # read_pfm 
    depth_map, _ = read_pfm(depth_path)
    print('depth shape: {}'.format(depth_map.shape))
    
    ## photometric filter
    #if False:
    #    pfm_prob_path = depth_path.replace("depth_est", "confidence")
    #    prob_map, _ = read_pfm(pfm_prob_path)
    #    depth_map[prob_map < 0.9] = 0
    
    # gray2color
    write_depth_img(depth_path.replace(".pfm", ".jpg"), depth_map)

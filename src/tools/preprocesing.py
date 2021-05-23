# -*- encoding: utf-8 -*-
import numpy as np
import copy
from scipy.ndimage.filters import gaussian_filter
import cv2


def im_to_double(im):
    """


    """
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    return (im.astype('float') - min_val) / (max_val - min_val)


def im_to_int(im):
    """

    """
    return (im.astype('int') * 255)


def ridge_segmentation(normalised_im, blksize, thresh):
    """


    """
    rows, cols = normalised_im.shape
    segmented_im = np.zeros((rows, cols))
    rows_block = int((rows / blksize) * blksize)
    cols_block = int((cols / blksize) * blksize)

    for i in range(0, rows_block, blksize):
        for j in range(0, cols_block, blksize):
            if (normalised_im[i:i + blksize,
                              j: j + blksize].var() >= thresh):
                segmented_im[i:i + blksize, j:j + blksize] = 1

    im1 = normalised_im - np.mean(normalised_im[np.where(segmented_im > 0)])
    stdh = np.std(im1[np.where(segmented_im > 0)])
    normalised_im = im1 / stdh

    return normalised_im, segmented_im


def ridge_orientation(im, orient_smooth_sigma):
    """

    """
    sobelx = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=3)
    
    orient = np.arctan2((sobely), (sobelx))
    
    Ox = np.cos(2 * orient)
    Oy = np.sin(2 * orient)
    
    sin2theta = gaussian_filter(Oy, orient_smooth_sigma, 0)
    cos2theta = gaussian_filter(Ox, orient_smooth_sigma, 0)
    
    return (np.arctan2(sin2theta, cos2theta) / 2)

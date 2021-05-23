# -*- encoding: utf-8 -*-
import numpy as np
import cv2


def ordfilt2(A, order):
    """

    """
    result = np.zeros((int(len(A)), 1))
    for n in range(0, int(len(A)), 1):
        if n + order < int(len(A)):
            band = A[range(n, n + order, 1)]
            result[n, 0] = band.max()
        else:
            band = A[range(n, int(len(A)), 1)]
            result[n, 0] = band.max()
    return result


def im_to_double(im):
    """

    """
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out


def rotate_image(image, angle):
    """

    """
    image_center = np.array(image.shape) / 2
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        image,
        rot_mat,
        image.shape,
        flags=cv2.INTER_LINEAR)
    return result


def frequency_estimation(
        im,
        orientim,
        window_size=3,
        min_wave_length=2,
        max_wave_length=15):
    """

    """
    cosorient = np.mean(np.cos(2 * np.array(orientim).ravel()))
    sinorient = np.mean(np.sin(2 * np.array(orientim).ravel()))
    
    orient = np.arctan2(sinorient, cosorient) / 2
    rotim = rotate_image(im, orient / np.pi * 180)
    
    rows, cols = im.shape
    
    crop_size = int(np.fix(rows / np.sqrt(2)))
    offset = int(np.fix((rows - crop_size) / 2))
    rotim = rotim[offset - 1: offset + crop_size +
                  1, offset - 1: offset + crop_size + 1]
    
    projection = rotim.sum(axis=0)
    dilation = ordfilt2(projection, window_size)
    
    maxpts = [x for x in range(len(dilation)) if np.logical_and(dilation[x] == projection[x], projection[x] > np.mean(projection))]
    
    if len(maxpts) < 2:
        freqim = np.zeros((rows, cols))
    else:
        number_of_peaks = len(maxpts)
        wave_length = (maxpts[number_of_peaks - 1] - maxpts[0]) / (number_of_peaks - 1)
        if max_wave_length > wave_length > min_wave_length:
            freqim = 1 / wave_length
        else:
            freqim = 0
        return freqim


def ridge_frequency(im, mask, orientim, blksize, window_size=3, minWL=2, maxWL=15):
    """

    """
    rows, cols = im.shape
    freq = np.zeros((len(im), len(im[0])))
    for r in range(0, rows - blksize, blksize):
        for c in range(0, cols - blksize, blksize):
            blkim = im[r: r + blksize,c: c + blksize]
            blkor = orientim[r: r + blksize, c: c + blksize]
            freq[r:r + blksize,c:c + blksize] = frequency_estimation(blkim,
                                                           blkor,
                                                           window_size,
                                                           minWL,
                                                           maxWL)
    ridge_freq_im = freq * mask
    median_freq = np.median(ridge_freq_im[np.where(ridge_freq_im > 0)])
    return ridge_freq_im, median_freq

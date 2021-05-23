# -*- encoding: utf-8 -*-

import cv2
import tools.preprocesing as preprocesing
import numpy as np
import tools.procesing as procesing

def main():

    image = cv2.imread("img/fp1.jpg", 0)

    # cv2.imshow("Original fingerprint", image)
    # cv2.waitKey(0)

    req_mean = 0
    req_var = 1
    thresh = 0.2
    mean_im = image.mean()
    std_im = image.std()
    im = image - mean_im
    im = im / std_im
    normalised_im = req_mean + im * np.sqrt(req_var)

    # cv2.imshow("Normalized fingerprint", normalised_im)
    # cv2.waitKey(0)

    normalised_im, work_mask_im = preprocesing.ridge_segmentation(
        normalised_im, 12, thresh)

    # cv2.imshow("Work zone fingerprint", work_mask_im)
    # cv2.waitKey(0)
    
    # cv2.imshow("Normalized on work zone fingerprint", normalised_im)
    # cv2.waitKey(0)

    orient_smooth_sigma = 5
    im_orientation = preprocesing.ridge_orientation(
        normalised_im, orient_smooth_sigma)

    # cv2.imshow("Orientation estimated fingerprint", im_orientation)
    # cv2.waitKey(0)

    blksize = 36
    freq, medfreq = procesing.ridge_frequency(normalised_im, work_mask_im, im_orientation, blksize)

    cv2.imshow("Frecuency estimated fingerprint",freq)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
import imageio
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import argparse


class ConverterImg():
    def __init__(self, url, name):
        self.url = url
        self.name = name

    def img_read(self):
        return imageio.imread(self.url)

    def grayscale(self, img):
        return np.dot(img[...,:3], [0.299, 0.587, 0.114])

    def gray_invert(self, img):
        return 255-img

    def blur_img(self, img):
        return scipy.ndimage.filters.gaussian_filter(img, sigma=5)

    def dodge(self, front, back):
        result = front * 255 / (255 - back)
        result[np.logical_or(result > 255, back == 255)] = 255
        return result.astype('uint8')

    def run(self):
        img = self.img_read()
        gray = self.grayscale(img)
        gray_inv = self.gray_invert(gray)
        blur_img = self.blur_img(gray_inv)

        result = self.dodge(blur_img, gray)
        plt.imsave("{}.png".format(self.name), result, cmap ="gray", vmin = 0, vmax = 255)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url', type=str,
                        help='imgs url')
    parser.add_argument('-n', '--name', type=str,
                        help='output file name')
    args = parser.parse_args()

    conv = ConverterImg(args.url, args.name)
    conv.run()

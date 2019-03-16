import numpy as np
from skimage.color import convert_colorspace, rgb2gray
from scipy.ndimage.filters import gaussian_filter
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage import img_as_ubyte
import camera_config

image_resize_x = camera_config.image_resize_x
image_resize_y = camera_config.image_resize_y


def grey_scale_histogram(grey_image_data):
    # # threshold grayscale
    # thresholded_img = img_gray > gray_scale_threshold

    # pre-process with difference of gaussian kernel
    blurred_img = gaussian_filter(grey_image_data, 5)
    edge_img = grey_image_data - blurred_img

    # plt.imshow(blurred_img)
    # plt.show()
    # plt.imshow(edge_img)
    # plt.show()

    gray_scale_hist, gray_scale_edges = np.histogram(edge_img.flatten(), 2 ** 8, (0, 255))
    return gray_scale_hist


def edge_features(rgb_image_data):
    # get greyscale image
    img_gray = rgb2gray(rgb_image_data)
    grey_8_bit = img_as_ubyte(img_gray)

    # plt.imshow(img_gray)
    # plt.show()

    histogram_features = grey_scale_histogram(grey_8_bit)
    return (histogram_features,)


def h_histogram_features(hsv_image_data):
    h_hist, h_hist_edges = np.histogram(hsv_image_data[:, :, 0].flatten(), 2 ** 4, (0, 1))
    return h_hist


def s_histogram_features(hsv_image_data):
    s_hist, s_hist_edges = np.histogram(hsv_image_data[:, :, 1].flatten(), 2 ** 4, (0, 1))
    return s_hist


def color_histogram_features(rgb_image_data):
    # change color space to HSV
    img_hsv = convert_colorspace(rgb_image_data, 'RGB', 'HSV')
    # plt.imshow(img_hsv)
    # plt.show()

    # hs color histogram
    h_hist = h_histogram_features(img_hsv)
    s_hist = s_histogram_features(img_hsv)

    return h_hist, s_hist


def resize_to_common_size(rgb_image_data):
    return resize(rgb_image_data, (image_resize_x, image_resize_y), anti_aliasing=True, mode="reflect")


def get_image_features(rgb_image_data):
    resized_image_data = resize_to_common_size(rgb_image_data)
    edge_feature = edge_features(resized_image_data)
    color_features = color_histogram_features(resized_image_data)
    return edge_feature + color_features

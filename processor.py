"""
Some functionality to process images. Note that basically everything shown
here can be done (better) using builtin functionality. This is done just
for fun and for learning purposes, not with efficiency/applicability in mind.

Some info is in wikipedia:
https://en.wikipedia.org/wiki/Digital_image_processing
"""

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import image


def _nearest(img: np.ndarray, size: tuple) -> np.ndarray:
    """
    Nearest neighbor interpolation
    :param img:
    :param size:
    :return:
    """
    new_r, new_c = size
    old_r, old_c = img.shape[:2]
    new_img = np.zeros(shape=(new_r, new_c, img.shape[-1]), dtype=np.uint8)
    for j in range(new_r):
        for i in range(new_c):
            r = math.floor(j / new_r * old_r)
            c = math.floor(i / new_c * old_c)
            new_img[j, i] = img[r, c]

    return new_img


def _bilinear(img: np.ndarray, size: tuple) -> np.ndarray:
    """
    Bilinear interpolation
    See https://en.wikipedia.org/wiki/Bilinear_interpolation
    :param img:
    :param size:
    :return:
    """
    new_r, new_c = size
    old_r, old_c = img.shape[:2]
    new_img = np.zeros(shape=(new_r, new_c, img.shape[-1]), dtype=np.uint8)
    for j in range(new_r):
        for i in range(new_c):
            x = i + .5
            y = j + .5
            x1 = math.floor(j / new_r * old_r)
            y1 = math.floor(i / new_c * old_c)
            x2 = x1 + 1
            y2 = y1 + 1
            dx1 = x - x1
            dx2 = x2 - x
            dy2 = y2 - y
            dy1 = y - y1
            f11 = img[x1, y1]
            f12 = img[x1, y2]
            f21 = img[x2, y1]
            f22 = img[x2, y2]
            f_arr = np.array([[f11, f12], [f21, f22]])
            a = 1 / ((x2 - x1) * (y2 - y1))
            x_arr = np.array([dx2, dx1])
            y_arr = np.array([dy2, dy1]).T
            new_img[j, i] = a * x_arr * np.dot(f_arr, y_arr)

    return new_img



def _bicubic(img: np.ndarray, size: tuple) -> np.ndarray:
    """
    Bicubic interpolation
    :param img:
    :param size:
    :return:
    """
    print("bicubic")


def resize(img: np.ndarray, scaler: int | float = None,
           size: tuple = None, method: str = "nearest") -> np.ndarray:
    """
    Resizes the image using either the given scalar or the size given
    in pixels. Either the "scaler" or the "pixels" parameter must be given,
    as both default to None. If both are given, the scaler will be used.
    :param img: The image to rescale (pixel colour scheme does not matter)
    :param scaler: An integer or float multiplier/scaler that will be applied
    to both axis. If < 1 the image will be downscaled, if > 1 the image will
    be upscaled. Preserves the aspect ratio.
    :param size: The exact size of the new image in pixels. The form of the
    tuple is (height, width).
    :param method: The interpolation method used to interpolate the pixel
    values. Options are "nearest", "bilinear", and "bicubic".
    :return:
    """
    if scaler is None and size is None:
        msg = "Either the 'scaler' or the 'size' parameter must be provided. " \
              "Now both were None."
        raise ValueError(msg)
    if scaler is None and len(size) > 2:
        msg = "Too many image dimensions provided. The form of the tuple must be " \
              "(height, width)."
        raise ValueError(msg)
    if scaler is None and len(size) < 2:
        msg = "Too few image dimensions provided. The form of the tuple must be " \
              "(height, width)."
        raise ValueError(msg)
    method_funs = {"nearest": _nearest, "bilinear": _bilinear,
                   "bicubic": _bicubic}
    if method not in method_funs.keys():
        ops = ", ".join(list(method_funs.keys()))
        msg = f"Invalid option for the interpolation method. Valid options " \
              f"are {ops}."
        raise ValueError(msg)
    if scaler is not None:
        h, w = img.shape[:2]
        scaled_size = math.ceil(scaler * h), math.ceil(scaler * w)
        return method_funs[method](img, scaled_size)
    if size is not None:
        return method_funs[method](img, size)


def get_component(img: np.ndarray, channel: int) -> np.ndarray:
    """
    Returns a single colour channel, i.e., a single component
    of the RGB values
    :param img: Da image itself in full RGB form
    :param channel: The colour channel to pick. The RGB channels are
        encoded in numbers such as R == 0, G == 1, and B == 2.
    :return:
    """
    if channel not in (0, 1, 2):
        msg = "Invalid colour channel. Valid options are 0 (red), 1 (green), " \
              "or 2 (blue)."
        raise ValueError(msg)
    new = np.zeros(shape=img.shape, dtype=np.uint8)
    chan = img[:, :, channel]
    new[:, :, channel] = chan
    return new


def grayscale(img: np.ndarray, method: str = "avg") -> np.ndarray:
    """
    Convert an RGB colour image into grayscale using the methods shown in
    https://www.baeldung.com/cs/convert-rgb-to-grayscale
    :param img:
    :param method: The way that the RGB value is converted into grayscale.
        The options are "avg", "lightness", and "luminosity".
    :return:
    """
    if method not in ["avg", "lightness", "weighted", "luminosity"]:
        msg = (f"The given grayscaling method '{method}' is invalid/not implemented yet. "
               f"Valid options are 'avg', 'lightness', 'weighted', and 'luminosity'.")
        raise ValueError(msg)
    gray_img = np.zeros(shape=img.shape[:-1], dtype=np.uint8)
    if method == "avg":
        gray_img[:, :] = np.mean(img, axis=2, keepdims=False)
        return gray_img
    if method == "lightness":
        mins = np.min(img, axis=2, keepdims=True)
        maxs = np.max(img, axis=2, keepdims=True)
        mins = mins.astype(np.uint16)
        maxs = maxs.astype(np.uint16)
        gray_img[:, :, :] = (mins + maxs) / 2
        return gray_img
    if method == "weighted":
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        z = .299 * r + .587 * g + .114 * b
        z.shape = (z.shape[0], z.shape[1], 1)
        gray_img[:, :, :] = z
        return gray_img
    if method == "luminosity":
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        z = .2126 * r + .7152 * g + .0722 * b
        z.shape = (z.shape[0], z.shape[1], 1)
        gray_img[:, :, :] = z
        return gray_img


def display_image_in_actual_size(img: np.ndarray) -> None:
    """
    Straight from
    https://stackoverflow.com/questions/28816046/displaying-different-images-with-
    actual-size-in-matplotlib-subplot
    :param img:
    :return:
    """
    dpi = mpl.rcParams['figure.dpi']
    height, width, depth = img.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes((0, 0, 1, 1))

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    plt.imshow(img)


def main() -> None:
    img = image.imread("./images/smol.jpg")
    if img.dtype != np.uint8:
        img = img[:, :, :3] * 255
        img = img.astype(np.uint8)
    img = grayscale(img=img, method="avg")
    print(img)
    plt.imshow(img)
    plt.title("Original")

    plt.figure()
    new = resize(img=img, size=(64, 64), method="bilinear")
    plt.title("Resized")
    # plt.imshow(new)
    display_image_in_actual_size(img=new)
    plt.show()


if __name__ == "__main__":
    main()

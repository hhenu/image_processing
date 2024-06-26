"""
Some functionality to process images or something. Quite inefficient
and probably somewhat incorrect solutions but it is what it is. Made for
learning purposes I guess instead of applicability and general usability.
"""

import math
import kernels
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import image
from typing import Any, Iterable


def load_img(img_path: str) -> np.ndarray:
    """
    Loads the given image file into a numpy array (in RGB) and with pixel
    values in range 0 ... 255
    :param img_path:
    :return:
    """
    img = image.imread(img_path)
    if img.dtype != np.uint8:
        img = img[:, :, :3] * 255
        img = img.astype(np.uint8)
    return img


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


def grayscale(img: np.ndarray, method: str = "avg", channels: int = 3) -> np.ndarray:
    """
    Convert an RGB colour image into grayscale using the methods shown in
    https://www.baeldung.com/cs/convert-rgb-to-grayscale
    :param img:
    :param method: The way that the RGB value is converted into grayscale.
    The options are "avg", "lightness", "weighted", and "luminosity".
    :param channels: The amount of colour channels to be returnd in the image.
    Can be either 1 or 3.
    :return:
    """
    methods = ["avg", "lightness", "weighted", "luminosity"]
    if method not in methods:
        msg = (f"The given grayscaling method '{method}' is invalid/not implemented. "
               f"Valid options are {", ".join(methods)}.")
        raise ValueError(msg)
    if channels not in (1, 3):
        msg = (f"Unsupported value for parameter channels {channels}. "
               f"Default value of 3 will be used.")
        print(msg)
        channels = 3
    gray_img = np.zeros(shape=img.shape, dtype=np.uint8)
    if method == "avg":
        gray_img[:, :, :] = np.mean(img, axis=2, keepdims=True)
    elif method == "lightness":
        mins = np.min(img, axis=2, keepdims=True)
        maxs = np.max(img, axis=2, keepdims=True)
        mins = mins.astype(np.uint16)
        maxs = maxs.astype(np.uint16)
        gray_img[:, :, :] = (mins + maxs) / 2
    elif method == "weighted":
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        z = .299 * r + .587 * g + .114 * b
        z.shape = (z.shape[0], z.shape[1], 1)
        gray_img[:, :, :] = z
    elif method == "luminosity":
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        z = .2126 * r + .7152 * g + .0722 * b
        z.shape = (z.shape[0], z.shape[1], 1)
        gray_img[:, :, :] = z
    else:
        assert False, "Unreachable"

    if channels == 3:
        return gray_img
    # Reduce the image to one color channel (e.g. the first one)
    return gray_img[:, :, 0]


def _nearest(img: np.ndarray, size: tuple) -> np.ndarray:
    """
    Nearest neighbor interpolation
    :param img:
    :param size:
    :return:
    """
    new_r, new_c = size
    old_r, old_c = img.shape[:2]
    try:
        new_shape = (new_r, new_c, img.shape[2])
    except IndexError:
        new_shape = (new_r, new_c)
    new_img = np.zeros(shape=new_shape, dtype=np.uint8)
    for j in range(new_r):
        for i in range(new_c):
            r = math.floor(j / new_r * old_r)
            c = math.floor(i / new_c * old_c)
            new_img[j, i] = img[r, c]

    return new_img


def _enumerate(seq: Iterable[Any], start: int = 0) -> tuple[int, Any]:
    """
    Pycharm seems to complain of the type hint of the built in 'enumerate()',
    so let's wrap it
    :param seq:
    :param start:
    :return:
    """
    for i, item in enumerate(seq, start=start):
        yield i, item


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
    x_arr = np.linspace(start=0, stop=old_c - 1, num=new_c, endpoint=False)
    y_arr = np.linspace(start=0, stop=old_r - 1, num=new_r, endpoint=False)
    try:
        chan = img.shape[2]
        new_img = np.zeros(shape=(new_r, new_c, chan), dtype=np.uint8)
    except IndexError:
        chan = 1
        new_img = np.zeros(shape=(new_r, new_c), dtype=np.uint8)
    # TODO: This coud use some refactoring I reckon
    if chan == 1:
        for j, y in _enumerate(y_arr):
            for i, x in _enumerate(x_arr):
                x1, x2 = math.floor(x), math.ceil(x)
                y1, y2 = math.floor(y), math.ceil(y)
                if x1 == x2 and x1 < old_c:
                    x2 += 1
                if y1 == y2 and y1 < old_r:
                    y2 += 1
                dx1 = x - x1
                dx2 = x2 - x
                dy1 = y - y1
                dy2 = y2 - y
                f11 = img[y1, x1]
                f12 = img[y2, x1]
                f21 = img[y1, x2]
                f22 = img[y2, x2]
                f_arr = np.array([[f11, f12], [f21, f22]])
                a = 1 / ((x2 - x1) * (y2 - y1))
                xx = np.array([dx2, dx1])
                yy = np.array([dy2, dy1]).T
                new_img[j, i] = a * np.dot(xx, np.dot(f_arr, yy))
    elif chan == 3:
        for j, y in _enumerate(y_arr):
            for i, x in _enumerate(x_arr):
                for c in range(chan):
                    x1, x2 = math.floor(x), math.ceil(x)
                    y1, y2 = math.floor(y), math.ceil(y)
                    if x1 == x2 and x1 < old_c:
                        x2 += 1
                    if y1 == y2 and y1 < old_r:
                        y2 += 1
                    dx1 = x - x1
                    dx2 = x2 - x
                    dy1 = y - y1
                    dy2 = y2 - y
                    f11 = img[y1, x1, c]
                    f12 = img[y2, x1, c]
                    f21 = img[y1, x2, c]
                    f22 = img[y2, x2, c]
                    f_arr = np.array([[f11, f12], [f21, f22]])
                    a = 1 / ((x2 - x1) * (y2 - y1))
                    xx = np.array([dx2, dx1])
                    yy = np.array([dy2, dy1]).T
                    new_img[j, i, c] = a * np.dot(xx, np.dot(f_arr, yy))
    else:
        raise ValueError(f"Invalid amount of colour channels {chan}")
    return new_img


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
    values. Options are "nearest" and "bilinear".
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
    method_funs = {"nearest": _nearest, "bilinear": _bilinear}
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


def conv(img: np.ndarray, kern: np.ndarray) -> np.ndarray:
    """
    Should convolute the image with the given kernel. Follows the pseudocode
    example from https://en.wikipedia.org/wiki/Kernel_(image_processing).
    Image edges are now handled by wrapping over the image.
    :param img:
    :param kern:
    :return:
    """
    k_row, k_col = kern.shape[:2]
    if len(kern.shape) > 2 or k_row != k_col:
        msg = f"Kernel must be a square 2d array, now got {k_row}x{k_col}"
        raise ValueError(msg)
    if k_row != 3 and k_row != 5:
        raise ValueError(f"Kernel must be either 3x3 or 5x5, now got {k_row}x{k_col}")
    r, c = img.shape[:2]
    try:
        chan = img.shape[2]
        new_img = np.zeros(shape=(r, c, chan), dtype=np.uint8)
    except IndexError:
        chan = 1
        new_img = np.zeros(shape=(r, c), dtype=np.uint8)
    dx, dy = k_col // 2, k_row // 2
    dx_r, dy_r = range(-dx, dx + 1), range(-dy, dy + 1)
    if chan == 1:
        for j in range(r):
            for i in range(c):
                acc = 0
                for k_j, k_row in _enumerate(seq=kern):
                    for k_i, k_val in _enumerate(seq=k_row):
                        dx, dy = dx_r[k_i], dy_r[k_j]
                        ind_x = (i + dx) % c
                        ind_y = (j + dy) % r
                        acc += k_val * img[ind_y, ind_x]

                new_img[j, i] = acc

    elif chan == 3:
        # TODO: Quite a bit of indentation here
        for j in range(r):
            for i in range(c):
                for ch in range(chan):
                    acc = 0
                    for k_j, k_row in _enumerate(seq=kern):
                        for k_i, k_val in _enumerate(seq=k_row):
                            dx, dy = dx_r[k_i], dy_r[k_j]
                            ind_x = (i + dx) % c
                            ind_y = (j + dy) % r
                            acc += k_val * img[ind_y, ind_x, ch]

                    new_img[j, i, ch] = acc

    else:
        raise ValueError(f"Unsupported amount of colour channels, {chan}")

    return new_img


def sobel(img: np.ndarray) -> np.ndarray:
    """
    Applies the Sobel operator to a grayscaled image

    See https://en.wikipedia.org/wiki/Sobel_operator
    :param img:
    :return:
    """
    if len(img.shape) > 2:
        msg = (f"Sobel operator is now only supported for 2d images, now got shape "
               f"{img.shape}")
        raise ValueError(msg)
    kern_x = kernels.sobel_x
    kern_y = kernels.sobel_y
    r, c = img.shape[:2]
    new_img = np.zeros(shape=(r, c))
    for j in range(r):
        for i in range(c):
            acc_x, acc_y = 0, 0
            for k_j, dy in enumerate(range(-1, 2)):
                for k_i, dx in enumerate(range(-1, 2)):
                    k_x = kern_x[k_j, k_i]
                    k_y = kern_y[k_j, k_i]
                    ind_x = (i + dx) % c
                    ind_y = (j + dy) % r
                    acc_x += k_x * img[ind_y, ind_x]
                    acc_y += k_y * img[ind_y, ind_x]

            new_img[j, i] = np.sqrt(acc_x * acc_x + acc_y * acc_y)

    return new_img


def display_in_actual_size(img: np.ndarray, title: str = None) -> None:
    """
    Straight from (note the link is in two lines):
    https://stackoverflow.com/questions/28816046/displaying-different-images-with-
    actual-size-in-matplotlib-subplot
    :param img:
    :param title:
    :return:
    """
    dpi = mpl.rcParams['figure.dpi']
    height, width = img.shape[:2]
    figsize = width / float(dpi), height / float(dpi)
    _ = plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)
    plt.imshow(img, cmap="gray")


def main() -> None:
    img_path = "./images/smol.jpg"
    img = load_img(img_path=img_path)
    img = grayscale(img=img, method="luminosity", channels=1)
    plt.imshow(img, cmap="gray")
    plt.title("Original")

    plt.figure()
    plt.title("Convoluted")
    new = sobel(img=img)
    plt.imshow(new, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()

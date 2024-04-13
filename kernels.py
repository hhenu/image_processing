"""
Couple of kernels for different types of convolutions

From https://en.wikipedia.org/wiki/Kernel_(image_processing)
"""

import numpy as np

ridge = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

edge = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

box_blur = np.ones(shape=(3, 3)) * (1 / 9)

gauss_blur3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) * (1 / 16)

gauss_blur5 = np.array([[1, 4, 6, 4, 1],
                        [4, 16, 24, 16, 4],
                        [6, 24, 36, 24, 6],
                        [4, 16, 24, 16, 4],
                        [1, 4, 6, 4, 1]]) * (1 / 256)

unsharp = np.array([[1, 4, 6, 4, 1],
                    [4, 16, 24, 16, 4],
                    [6, 24, -476, 24, 6],
                    [4, 16, 24, 16, 4],
                    [1, 4, 6, 4, 1]]) * (-1 / 256)

sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

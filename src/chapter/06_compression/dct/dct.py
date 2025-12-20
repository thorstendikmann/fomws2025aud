# type hint for classes, see https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class
import sys
from tkinter import Y
import numpy as np
import math
import scipy.fftpack
import logging
logger = logging.getLogger(__name__)


def dct(values: np.array) -> np.array:
    """Calculates the discrete cosine transformation for given array.

    Args:
        values (np.array): input array.

    Returns:
        np.array: discrete cosine transformation of the input.
    """
    returnArray = np.array(values, dtype=float)
    n = len(values)

    for count, val in enumerate(values):
        logger.debug(f"{count}:")
        if (count == 0):
            X_0 = 0
            for i, tmpVal in enumerate(values):
                X_0 += tmpVal
                logger.debug(f"    X_0 (tmp {i}) = {X_0}")
            X_0 = math.sqrt(1/n) * X_0
            logger.debug(f"  X_0 = {X_0}")
            returnArray[0] = X_0
        else:
            X_k = 0
            for i, x_i in enumerate(values):
                X_k += x_i * math.cos((math.pi / n)
                                      * (i + 1/2) * count)
                logger.debug(
                    f"    X_{count} (tmp {i}) = {X_k} |   x_i={x_i}, Pi/{n}={math.pi / n}, {i}+1/2*{count}={(i + 1/2) * count} | cos({(math.pi / n)* (i + 1/2) * count}) = {math.cos((math.pi / n) * (i + 1/2) * count)}")
            X_k = math.sqrt(2/n) * X_k
            logger.debug(f"  X_{count} = {X_k}")
            returnArray[count] = X_k

    return returnArray


def idct(values: np.array, max_steps=sys.maxsize, min_step=0) -> np.array:
    """Calculates the inverse discrete cosine transformation for given array.

    Use <tt>max_steps</tt> to simulate approximation. If max_steps < len(values),
    this array will return the original function only to a certain degree.
    Use <tt>min_step</tt> in combination with <tt>max_steps</tt> to isolate
    individual harmonics.

    Args:
        values (np.array): input of dct array.
        max_steps (int): limit of terms to be evaluated in calculating every x_k.
        min_step (int): Start iteration at given index.  x_k.

    Returns:
        np.array: sample points of original function.
    """
    returnArray = np.array(values, dtype=float)
    n = len(values)

    for count, val in enumerate(values):
        x_k = 0
        for i, x_i in enumerate(values):
            if i >= max_steps:
                break
            if i < min_step:
                continue
            if i == 0:
                continue
            x_k += x_i * math.cos((math.pi / n) * i * (count + 1/2))
        x_k = (math.sqrt(1/n) * values[0]) + (math.sqrt(2/n) * x_k)
        returnArray[count] = x_k

    return returnArray


def quantizationBits(values: np.array) -> np.array:
    """
    Quantization function for DCT.
    This will cut away the lower bits of each value in given values array.

    Args:
        values (np.array): DCT input values.

    Returns:
        np.array: Quantized array.
    """
    quantum = np.array(values, dtype=int)
    for i, q in enumerate(quantum):
        q &= ~0b1111
        quantum[i] = q

    return quantum


# Some quantization array for below
quantum = np.array([3, 3, 3, 4, 4, 5, 5, 6,
                    7, 7, 8, 8, 9, 9, 9, 10])


def quantizationWeight(values: np.array) -> np.array:
    """
    Quantization function for DCT.
    This will quantize each value with a given matrix.

    Args:
        values (np.array): DCT input values.

    Returns:
        np.array: Quantized DCT array.
    """

    # This is the "original Q50 array for JPEG compression
    # Q50 = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
    #                 [12, 12, 14, 19, 26, 48, 60, 55],
    #                 [14, 13, 16, 24, 40, 57, 69, 56],
    #                 [14, 17, 22, 29, 51, 87, 80, 62],
    #                 [18, 22, 37, 56, 68, 109, 103, 77],
    #                 [24, 35, 55, 64, 81, 104, 113, 92],
    #                 [49, 64, 78, 87, 103, 121, 120, 101],
    #                 [72, 92, 95, 98, 112, 100, 103, 99]])

    returnArray = np.array(values / quantum, dtype=int)
    return returnArray


def deQuantizationWeight(values: np.array) -> np.array:
    """
    Inverse of quantizationWeight function.

    Args:
        values (np.array): DCT input values as quantized by the quantizationWeight function.

    Returns:
        np.array: De-Quantized DCT array.
    """
    returnArray = np.array(values * quantum, dtype=int)
    return returnArray


if __name__ == '__main__':
    """main guard for module."""
    # print("No main function in here.")
    data = np.array([10, 20, 50, 200, 180, 150, 100, 50, 10,
                     75, 95, 105, 135, 180, 210, 240, 255, 230, 200, 100])

    dct_data = dct(data)
    dct_data_scipy = scipy.fftpack.dct(data, norm='ortho')

    idct_data = idct(dct_data, max_steps=19)
    icdct_data_scipy = scipy.fftpack.idct(dct_data, norm='ortho')

    print(data)
    print("########## SciPy")
    print(dct_data_scipy)
    print(icdct_data_scipy)
    print("########## Own")
    print(dct_data)
    print(idct_data)

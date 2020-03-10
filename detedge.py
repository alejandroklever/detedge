from typing import Tuple, Callable, Union

import numpy as np


##############
# Type Hints #
##############
Mask = np.ndarray
Spectrum = np.ndarray
Range = Tuple[float, float]
FFilter = Callable[[np.ndarray, Union[float, Range]], np.ndarray]


def lpf(img: np.ndarray, r: float) -> Mask:
    """
    Band Pass Filter compute a binary mask where
    mask[i, j] = 1 if
        - (i - x0) ^ 2 + (j - y0) ^ 2 <= r ^ 2
    where x0 and y0 are thr coords of the center of the image

    :param img: image as 2-D numpy.ndarray
    :param r: radio for clean the high frequencies
    :return: the binary mask
    """
    rows, cols = img.shape
    center = (rows // 2, cols // 2)

    mask = np.zeros((rows, cols), np.uint8)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r ** 2
    mask[mask_area] = 1

    return mask


def hpf(img: np.ndarray, r: float) -> Mask:
    """
    Band Pass Filter compute a binary mask where
    mask[i, j] = 0 if
        - (i - x0) ^ 2 + (j - y0) ^ 2 <= r ^ 2
    where x0 and y0 are thr coords of the center of the image

    :param img: image as 2-D numpy.ndarray

    :param r: radio for clean the high frequencies

    :return: the binary mask
    """

    rows, cols = img.shape
    center = (rows // 2, cols // 2)

    mask = np.ones((rows, cols), np.uint8)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r ** 2
    mask[mask_area] = 0

    return mask


def bpf(img: np.ndarray, r: Tuple[float, float]) -> Mask:
    """
    Band Pass Filter compute a binary mask where
    mask[i, j] = 1 if
        - r2 ^ 2 >= (i - x0) ^ 2 + (j - y0) ^ 2 >= r ^ 2
    where x0 and y0 are thr coords of the center of the image

    :param img: image as 2-D numpy.ndarray

    :param r: 2-items tuple (inner_radio, outer_radio)

    :return: a 2-D binary mask as numpy.ndarray
    """

    r1, r2 = r
    rows, cols = img.shape
    center = (rows // 2, cols // 2)

    mask = np.zeros((rows, cols), np.uint8)
    x, y = np.ogrid[:rows, :cols]
    mask_area = np.logical_and(
        ((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r1 ** 2),
        ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r2 ** 2)
    )

    print(mask_area.shape)
    mask[mask_area] = 1

    return mask


def detect_edges(img: np.ndarray, ffilter: FFilter, r: Union[float, Range]) -> Tuple[np.ndarray, Tuple[Spectrum, Mask]]:
    """
    Compute fft in the given image and filter the frequencies using the mask
    returned by ffilter function in the given radio or radio_range

    :param img: image as 2-D numpy.ndarray

    :param ffilter: one of the functions lpf, hpf, bpf

    :param r: if ffilter is lpf or hpf is a float representing the radio
            if ffilter is bpf is 2-items tuple with (inner_radio, outer_radio)

    :return: image as numpy.array with the applied filter
    """
    
    mask: np.ndarray = ffilter(img, r)
    fft: np.ndarray = np.fft.fft2(np.float32(img))
    fft_shift: np.ndarray = np.fft.fftshift(fft)

    magnitude_spectrum: np.ndarray = 20 * np.log(np.abs(fft_shift))

    # apply a mask and inverse DFT
    fft_shift *= mask

    fshift_mask_mag = 2000 * np.log(np.abs(fft_shift))

    ifft_shift = np.fft.ifftshift(fft_shift)
    img_back = np.abs(np.fft.ifft2(ifft_shift))

    return img_back, (magnitude_spectrum, fshift_mask_mag)

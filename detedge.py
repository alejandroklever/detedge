from typing import Tuple

import numpy as np


def hpf(image: np.ndarray, radio: float) -> np.ndarray:
    rows, cols = image.shape
    center = (rows // 2, cols // 2)

    mask = np.ones((rows, cols), np.uint8)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radio ** 2
    mask[mask_area] = 0

    return mask


def lpf(image: np.ndarray, radio: float) -> np.ndarray:
    rows, cols = image.shape
    center = (rows // 2, cols // 2)

    mask = np.zeros((rows, cols), np.uint8)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radio ** 2
    mask[mask_area] = 1

    return mask


def bpf(image: np.ndarray, radio: float, radio2: float) -> np.ndarray:
    rows, cols = image.shape
    center = (rows // 2, cols // 2)

    mask = np.zeros((rows, cols), np.uint8)
    x, y = np.ogrid[:rows, :cols]
    mask_area = np.logical_and(
        ((x - center[0]) ** 2 + (y - center[1]) ** 2 >= radio ** 2),
        ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= radio2 ** 2)
    )
    mask[mask_area] = 1

    return mask


def detect_edges(image: np.ndarray, freq_filter, radio: float, radio2: float = None) \
        -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    if radio2 is not None:
        mask: np.ndarray = freq_filter(image=image, radio=radio, radio2=radio2)
    else:
        mask: np.ndarray = freq_filter(image=image, radio=radio)

    fft: np.ndarray = np.fft.fft2(np.float32(image))
    fft_shift: np.ndarray = np.fft.fftshift(fft)

    magnitude_spectrum: np.ndarray = 20 * np.log(np.abs(fft_shift))

    # apply a mask and inverse DFT
    fft_shift *= mask

    fshift_mask_mag = 2000 * np.log(np.abs(fft_shift))

    ifft_shift = np.fft.ifftshift(fft_shift)
    img_back = np.abs(np.fft.ifft2(ifft_shift))

    return img_back, (magnitude_spectrum, fshift_mask_mag)

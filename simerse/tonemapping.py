
import numpy as np


class Reinhard:
    def __init__(self, white_point_quantile=.9):
        self.quantile = white_point_quantile

    def __call__(self, im):
        lum = 0.2126 * im[:, :, 0] + 0.7152 * im[:, :, 1] + 0.0722 * im[:, :, 2]
        lum_value = np.quantile(lum.flatten(), self.quantile)
        print(lum_value)
        lum_out = lum * (1 + lum / lum_value ** 2) / (1 + lum)
        im *= (lum_out / lum).reshape(*im.shape[0:2], 1)
        return im

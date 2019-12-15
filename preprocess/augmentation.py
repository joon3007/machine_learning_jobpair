import scipy.misc
from scipy.ndimage import zoom
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import colorsys



# ref site : https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv/30609854
def __noisy(img, noise_type = 'gaussian', mean = 0, var = 10):
# Gaussian-distributed additive noise
    if noise_type == 'gaussian':
        row, col, ch = img.shape
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = img + gauss
        return noisy
        # Replace random pixels with 0 or 1
    elif noise_type == 's&p':
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(img)

        # salt mode
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
        out[coords] = 1

        # pepper mode
        num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
        out[coords] = 0
        return out
        # Poisson-distributed noise generated from the data
    elif noise_type == 'poisson':
        vals = len(np.unique(img))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(img * vals) / float(vals)
        return noisy
    # Multiplicative noise using out = image + n*image, where n is uniform noise with specified mean & variance
    elif noise_type == 'speckle':
        row, col, ch = img.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = img + img * gauss
        return noisy

# add gaussian noise
def gaussian_noise(img, mean = 0, var = 10):
    return __noisy(img, noise_type = 'gaussian', mean = mean, var = var)

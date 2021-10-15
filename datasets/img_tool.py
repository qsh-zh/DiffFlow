# pylint: skip-file
import numpy as np
from jammy.image import imread


def prepare_image(
    img_path, crop=None, embed=None, white_cutoff=225, gauss_sigma=5, background=0.0001
):
    """Transforms rgb image array into 2D-density and energy

    Parameters
    ----------
    density : ndarray(width, height)
        Probability density

    energy : ndarray(width, height)
        Energy

    """
    img = imread(img_path)

    # make one channel
    img = img.mean(axis=2)

    # make background white
    img = img.astype(np.float32)
    img[img > white_cutoff] = 255

    # normalize
    img /= img.max()

    if crop is not None:
        # crop
        img = img[crop[0] : crop[1], crop[2] : crop[3]]

    if embed is not None:
        tmp = np.ones((embed[0], embed[1]), dtype=np.float32)
        shift_x = (embed[0] - img.shape[0]) // 2
        shift_y = (embed[1] - img.shape[1]) // 2
        tmp[shift_x : img.shape[0] + shift_x, shift_y : img.shape[1] + shift_y] = img
        img = tmp

    # convolve with Gaussian
    from scipy.ndimage import gaussian_filter

    # TODO: need to tune the gauss_sigma smoothness of image
    img2 = gaussian_filter(img, sigma=gauss_sigma)

    # add background
    background1 = gaussian_filter(img, sigma=10)
    background2 = gaussian_filter(img, sigma=20)
    background3 = gaussian_filter(img, sigma=50)
    density = (1.0 - img2) + background * (background1 + background2 + background3)

    return density


class ImageSampler(object):
    def __init__(self, img_density, mean=[350, 350], scale=[350, 350]):
        """Samples continuous coordinates from image density

        Parameters
        ----------
        img_density : ndarray(width, height)
            Image probability density

        mean : (int, int)
            center pixel

        scale : (int, int)
            number of pixels to scale to 1.0 (in x and y direction)

        """
        self.img_density = img_density
        Ix, Iy = np.meshgrid(
            np.arange(img_density.shape[1]), np.arange(img_density.shape[0])
        )
        self.idx = np.vstack([Ix.flatten(), Iy.flatten()]).T

        # draw samples from density
        density_normed = img_density.astype(np.float64)
        density_normed /= density_normed.sum()
        self.density_flat = density_normed.flatten()
        self.mean = np.array([mean])
        self.scale = np.array([scale])

    def sample(self, nsample):
        # draw random index
        i = np.random.choice(self.idx.shape[0], size=nsample, p=self.density_flat)
        ixy = self.idx[i, :]

        # simple dequantization, uniformally sample in the grid
        xy = ixy + np.random.rand(nsample, 2) - 0.5

        # normalize shape
        xy = (xy - self.mean) / self.scale

        return xy

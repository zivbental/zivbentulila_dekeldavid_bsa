# example python script for loading neurofinder data
#
# for more info see:
#
# - http://neurofinder.codeneuro.org
# - https://github.com/codeneuro/neurofinder
#
# requires three python packages
#
# - numpy
# - tifffile
# - matplotlib
#

import json
import matplotlib.pyplot as plt
from numpy import array, zeros
from tifffile import imread
from glob import glob


def tomask(coords):
    """Create a masked version of the image."""
    mask = zeros(dims)
    coords = array(coords)
    mask[coords[:, 0], coords[:, 1]] = 1
    return mask


if __name__ == "__main__":
    # load the images
    files = sorted(glob('images/*.tiff'))
    imgs = imread(files)
    dims = imgs.shape[1:]

    # load the regions (training data only)
    with open('regions/regions.json') as f:
        regions = json.load(f)

    masks = array([tomask(s['coordinates']) for s in regions])

    # show the outputs
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(imgs.sum(axis=0), cmap='gray')
    axes[1].imshow(masks.sum(axis=0), cmap='gray')
    plt.show()

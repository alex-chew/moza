import glob
import math
import random

import numpy as np
from PIL import Image

TILES_DIR = 'tiles'
SOURCE_IMG = 'source.jpg'
TARGET_IMG = 'target.jpg'
BLOCK_SIZE = 20
TILE_CHOICES = 3


def color_dist(c1, c2):
    '''Weighted RGB color distance to fit human perception, from
    <https://en.wikipedia.org/wiki/Color_difference>.
    '''
    dr2, dg2, db2 = (c1 - c2)**2
    avg_r = (c1[0] + c2[0]) / 2
    return (2 * dr2) + (4 * dg2) + (3 * db2) + (avg_r * (dr2 - db2) / 256)


def main():
    # Reformat image as array
    source_img = Image.open(SOURCE_IMG)
    source_arr = np.transpose(source_img, (1, 0, 2))
    blocks_x = math.ceil(source_img.width / BLOCK_SIZE)
    blocks_y = math.ceil(source_img.height / BLOCK_SIZE)

    # Compute block average colors
    block_colors = np.zeros((blocks_x, blocks_y, 3))
    for c, block_col in enumerate(np.split(source_arr, blocks_x, axis=0)):
        for r, block in enumerate(np.split(block_col, blocks_y, axis=1)):
            block_colors[c, r] = block.mean(axis=(0, 1))

    # Compute tile average colors
    tile_images = {
        path: np.asarray(Image.open(path))
        for path in glob.glob('{}/*'.format(TILES_DIR))
    }
    tile_colors = [(img.mean(axis=(0, 1)), path)
                   for path, img in tile_images.items()]

    # Assemble target by selecting a similarly-colored tile for each block
    rows = []
    for row_colors in block_colors:
        row_tiles = []
        for block_color in row_colors:
            sort_key = lambda c2_path: color_dist(block_color, c2_path[0])
            closest_tiles = sorted(tile_colors, key=sort_key)[:TILE_CHOICES]
            tile_path = random.choice(closest_tiles)[1]
            row_tiles.append(tile_images[tile_path])
        rows.append(np.concatenate(row_tiles, axis=0))
    target = np.concatenate(rows, axis=1)

    # Output image
    Image.fromarray(target).save(TARGET_IMG)


if __name__ == '__main__':
    main()

import glob
import math
import random

import click
import numpy as np
from PIL import Image


def color_dist(c1, c2):
    """
    Weighted RGB color distance to fit human perception, from
    <https://en.wikipedia.org/wiki/Color_difference>.
    """
    dr2, dg2, db2 = (c1 - c2)**2
    avg_r = (c1[0] + c2[0]) / 2
    return (2 * dr2) + (4 * dg2) + (3 * db2) + (avg_r * (dr2 - db2) / 256)


def get_block_colors(array, block_size):
    """
    Reduces an array by averaging all colors in each square block.

    :param array: the source array
    :param block_size: the maximum side length of each block, in pixels
    :returns: an array representing the average color of each corresponding
        block in the source array
    """
    blocks_x = math.ceil(array.shape[0] / block_size)
    blocks_y = math.ceil(array.shape[1] / block_size)
    block_colors = np.zeros((blocks_x, blocks_y, array.shape[2]))

    for c, block_col in enumerate(np.array_split(array, blocks_x, axis=0)):
        for r, block in enumerate(np.array_split(block_col, blocks_y, axis=1)):
            block_colors[c, r] = block.mean(axis=(0, 1))
    return block_colors


def get_tiles(paths):
    """
    Returns a dict of format `{path: arr}` where `arr` contains the tile data,
    and a dict of format `{path: color}` where `color` is the average color of
    the tile.

    :param paths: iterable of tiles' paths
    :returns: `(images_dict, colors_dict)` as described above
    """
    images = {path: np.asarray(Image.open(path)) for path in paths}
    colors = {path: images[path].mean(axis=(0, 1)) for path in paths}
    return images, colors


@click.command()
@click.argument('source', type=click.Path(exists=True, dir_okay=False))
@click.argument('tiles', type=click.Path(exists=True, file_okay=False))
@click.argument('target', type=click.Path(dir_okay=False, writable=True))
@click.option(
    '--blocksize', '-b', default=20, help='size of source blocks, in pixels')
@click.option(
    '--choices',
    '-c',
    default=3,
    help='number of similarly-colored tiles to choose from')
def assemble_mosaic(source, tiles, target, blocksize, choices):
    # Get tiles, tile colors, source, and source block colors
    tile_images, tile_colors = get_tiles(glob.glob('{}/*'.format(tiles)))
    source_image = np.transpose(Image.open(source), (1, 0, 2))
    block_colors = get_block_colors(source_image, blocksize)

    click.echo('Creating {}-tile-by-{}-tile mosaic, selecting from {} tiles.'
               .format(block_colors.shape[0], block_colors.shape[1],
                       len(tile_images)))

    # Assemble target by selecting a similarly-colored tile for each block
    rows = []
    for row_colors in block_colors:
        row_tiles = []
        for block_color in row_colors:
            sort_key = lambda path_c2: color_dist(block_color, path_c2[1])
            closest_tiles = sorted(tile_colors.items(), key=sort_key)[:choices]
            tile_path = random.choice(closest_tiles)[0]
            row_tiles.append(tile_images[tile_path])
        rows.append(np.concatenate(row_tiles, axis=0))
    target_image = np.concatenate(rows, axis=1)

    # Output image
    Image.fromarray(target_image).save(target)

    click.echo('{}x{} mosaic saved as {}.'
               .format(target_image.shape[0], target_image.shape[1], target))


if __name__ == '__main__':
    assemble_mosaic()

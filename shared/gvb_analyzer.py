import argparse
import re
from pathlib import Path

import numpy as np
import skimage
import tifffile as tiff
from natsort import natsorted
from tqdm import tqdm


def correct_all(data_dir, output_dir):

    # Read all files in
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    elif isinstance(data_dir, Path):
        pass
    else:
        raise TypeError(
            f"Expected data directory to be a str or Path. Instead it has type {type(data_dir)}."
        )

    if not data_dir.exists():
        raise FileNotFoundError(f"The directory {data_dir} does not exist.")

    # Read all files in
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    elif isinstance(output_dir, Path):
        pass
    else:
        raise TypeError(
            f"Expected output directory to be a str or Path. Instead it has type {type(output_dir)}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    image_list = list(data_dir.glob("*.TIF"))

    pattern = re.compile(r"(?P<base>.*)_s(?P<tile>\d+)_w(?P<channel>\d+)\.TIF")

    base = set()
    tiles = set()
    channels = set()

    for f in image_list:
        match = pattern.search(f.name)

        if match:
            base.add(match.group("base"))
            tiles.add(int(match.group("tile")))
            channels.add(int(match.group("channel")))

    tiles = sorted(tiles)
    channels = sorted(channels)

    print(f"Number of tiles: {len(tiles)}.")
    print(f"Number of channels: {len(channels)}.")

    if len(base) > 1:
        raise ValueError(f"Expected only 1 base filename but found {len(base)}.")

    base = list(base)

    # Obtain the shading for each channel
    for iC in range(len(channels)):
        correct_shading(
            data_dir,
            output_dir / (base[0] + f"_channel{iC + 1}"),
            file_pattern=f"*_w{iC + 1}.TIF",
        )

    # Correct each channel

    # Get the stitching coordinates using the DAPI channel

    # Stitch them all together

    # Register the stitched images to each other


def calculate_shading(data_dir, file_pattern="*_w2.TIF", return_image_stack=True):
    """
    Calculate shading for each tile.

    The tiles are stacked on each other, and the median intensity is used to estimate
    the shading pattern. The median intensity is Gaussian filtered with a sigma of 50 to
    obtain the final shading pattern.

    Parameters
    ----------
    data_dir : str or Path
        Directory to images
    file_pattern : str, optional
        Filename pattern to search for, by default "*_w2.TIF"
    return_image_stack : bool, optional
        If True, returns the images as an 3D ndarray, by default True

    Returns
    -------
    shading : ndarray of float64
        Estimated shading pattern, retaining the original intensity scale
    image_data : ndarray
        3D ndarray with images stacked along the third dimension, with the same data
        type as the original images

    Raises
    ------
    FileNotFoundError
        The input data_dir does not exist
    ValueError
        The input data_dir is not a valid directory
    FileNotFoundError
        No files matching file_pattern was found in data_dir
    """

    # Validate the inputs
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"The directory {data_dir.resolve()} does not exist.")
    elif not data_dir.is_dir():
        raise ValueError(
            f"The input {data_dir} does not appear to be a valid directory."
        )

    # Find images and sort using natural sorting (this order is important otherwise "10"
    # get sorted before "1" and "2").
    image_list = list(data_dir.glob(file_pattern))

    if len(image_list) == 0:
        raise FileNotFoundError(
            f"No files matching the pattern '{file_pattern}' was found in {data_dir}."
        )

    image_list = natsorted(image_list)

    # Read an image to get its size and shape
    image = skimage.io.imread(image_list[0])

    # Create a matrix to hold the images
    image_data = np.zeros(
        (image.shape[0], image.shape[1], len(image_list)),
        dtype=image.dtype,
    )

    # Load in the images
    for idx, file in enumerate(tqdm(image_list, desc="Reading images")):
        image_data[:, :, idx] = skimage.io.imread(file)

    # Compute the median to get the shading
    shading = np.median(image_data, axis=-1)

    # Blur the resulting data
    shading = skimage.filters.gaussian(shading, sigma=50)

    print(shading.dtype)
    exit()

    # plt.imshow(shading)
    # plt.show()

    if return_image_stack:
        return shading, image_data

    else:
        return shading


def correct_shading(data_dir, output_dir, shading=None, **kwargs):
    """
    Corrects shading.

    This function estimates the shading from a series of tiles, then writes the shading
    corrected tiles to a folder. The images will be named correctly to be stitched using
    Fiji/ImageJ's Image Stitching package.

    Parameters
    ----------
    data_dir : str or Path
        Path to directory containing image tiles
    output_dir : str or Path
        Path to directory to output corrected images
    shading : ndarray, optional
        Array of estimated or measured shading, by default None. If None, the shading
        will be estimated from the tiles.
    """

    # Validate the inputs
    if shading is None:
        shading, image_data = calculate_shading(data_dir, **kwargs)

    output_dir = Path(output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Get the original data type and maximum integer value
    original_dtype = image_data.dtype
    max_val = (np.iinfo(original_dtype)).max

    # Convert the images from integer to float32 for division
    image_data = image_data.astype(np.float32)

    # Calculate the mean shading to restore values
    shading_mean = np.mean(shading)

    for ii in tqdm(range(image_data.shape[-1]), desc="Saving corrected images"):
        curr_image = image_data[:, :, ii]
        curr_image = (curr_image / shading) * shading_mean

        # Return image back to uint16
        curr_image = np.clip(curr_image, 0, max_val)

        skimage.io.imsave(
            output_dir / f"img_{(ii + 1):02}.tif",
            curr_image.astype(original_dtype),
            check_contrast=False,
        )

    # Save the shading information
    skimage.io.imsave(output_dir / "shading.tiff", shading)


def merge_tiffs(image_dir_list, output_dir, channel_names=None):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_list = []

    for d in image_dir_list:
        image_list.extend(Path(d).glob("*.tif"))

    images = [skimage.io.imread(f) for f in image_list]

    try:
        stack = np.stack(images, axis=0)
    except Exception:
        print("Images are not the same sizes.")
        for f, img in zip(image_list, images):
            print(f"{f}: shape = {img.shape}, dtype = {img.dtype}")
        exit()

    # num_levels = 3
    # pyramid = [stack]

    # for level in range(1, num_levels):
    #     prev_h, prev_w, num_c = pyramid[-1].shape
    #     new_h, new_w = prev_h // 2, prev_w // 2

    #     # Resize each channel independently to preserve exact spatial scaling and prevent channel-axis distortion
    #     scaled_channels = [
    #         skimage.transform.resize(
    #             pyramid[-1][..., c],
    #             (new_h, new_w),
    #             anti_aliasing=True,
    #             preserve_range=True,
    #             order=1,
    #         ).astype(stack.dtype)
    #         for c in range(num_c)
    #     ]
    #     # Re-stack back to YXC format
    #     scaled_img = np.stack(scaled_channels, axis=-1)
    #     pyramid.append(scaled_img)

    if channel_names is None:
        channel_names = [f"Channel {i + 1}" for i in range(stack.shape[0])]

    # print(channel_names)
    # exit()

    tile_size = 256
    subresolutions = 2

    with tiff.TiffWriter(output_dir / "combined.ome.tif", bigtiff=True) as tif:
        options = {
            "photometric": "minisblack",
            "tile": (tile_size, tile_size),
            "compression": "lzw",
            "resolutionunit": "CENTIMETER",
        }

        tif.write(
            stack,
            subifds=subresolutions,
            metadata={"axes": "CYX", "Channel": {"Name": channel_names}},
            **options,
        )

        for level in range(subresolutions):
            mag = 2 ** (level + 1)
            tif.write(
                stack[:, ::mag, ::mag],
                subfiletype=1,  # FILETYPE.REDUCEDIMAGE
                **options,
            )


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description="Estimate and correct shading across a directory of image tiles."
    )

    # Required arguments
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Path to the directory containing raw input images.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Path to save corrected output images and shading map.",
    )

    # Optional arguments
    parser.add_argument(
        "-p",
        "--pattern",
        type=str,
        default="*.TIF",
        help="File pattern/extension to match (default: '*.TIF').",
    )

    args = parser.parse_args()

    # Execute main workflow
    correct_shading(
        data_dir=args.input_dir, file_pattern=args.pattern, output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

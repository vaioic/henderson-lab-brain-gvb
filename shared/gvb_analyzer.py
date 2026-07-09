import re
from pathlib import Path

import numpy as np
import skimage
from matplotlib import pyplot as plt
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

    pass


def calculate_shading(data_dir, file_pattern="*_w2.TIF", return_image_stack=True):

    print(file_pattern)

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

    plt.imshow(shading)
    plt.show()

    if return_image_stack:
        return shading, image_data

    else:
        return shading


def correct_shading(data_dir, output_dir, shading=None, **kwargs):
    # This function corrects the shading and output the images, named correctly to be
    # stitched in Fiji.

    if shading is None:
        shading, image_data = calculate_shading(data_dir, **kwargs)

    # Save the images
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Get the original data type
    original_dtype = image_data.dtype
    # print(original_dtype)

    max_val = (np.iinfo(original_dtype)).max

    # Correct the images
    image_data = image_data.astype(np.float32)

    shading_mean = np.mean(shading)

    for ii in tqdm(range(image_data.shape[-1]), desc="Saving corrected images"):
        curr_image = image_data[:, :, ii]

        # plt.imshow(curr_image)
        # plt.show()
        # plt.close()

        curr_image = (curr_image / shading) * shading_mean

        # plt.imshow(curr_image)
        # plt.show()
        # plt.close()

        # Return image back to uint16
        curr_image = np.clip(curr_image, 0, max_val)

        skimage.io.imsave(
            output_dir / f"img_{(ii + 1):02}.tif",
            curr_image.astype(original_dtype),
            check_contrast=False,
        )

    # Save the shading information
    skimage.io.imsave(output_dir / "shading.tiff", shading)


# def register_images(target, moving):

#     h0, w0 = target.shape
#     hm, wm = moving.shape

#     # Match the moving to the target shape
#     moving_final = np.zeros_like(target)

#     # Determine the overlapping bounds
#     slice_y = min(h0, hm)
#     slice_x = min(w0, wm)

#     moving_final[:slice_y, :slice_x] = moving[:slice_y, :slice_x]

#     # Run the cross-correlation to register the images
#     shift, error, diffphase = sk.registration.phase_cross_correlation(
#         target,
#         moving_final,
#         disambiguate=True
#     )

#     moving_corrected = ndimage_shift(
#         moving_final,
#         shift=shift,
#         cval=0.0
#     )

#     plot_merged_images(target, moving_corrected)


# def plot_merged_images(image1, image2):

#     image1 = sk.exposure.rescale_intensity(image1, out_range=(0.0, 1.0))
#     image2 = sk.exposure.rescale_intensity(image2, out_range=(0.0, 1.0))

#     merged = np.zeros((image1.shape[0], image1.shape[1], 3), dtype=np.uint8)

#     merged[..., 0] = sk.util.img_as_ubyte(image1)
#     merged[..., 1] = sk.util.img_as_ubyte(image2)
#     merged[..., 2] = sk.util.img_as_ubyte(image1)

#     plt.imshow(merged)
#     plt.show()


# I1 = sk.io.imread("../data/Dataset 1/round 001/AW GVB AM1c-s11 010426_A01_w2.tif")

# I2 = sk.io.imread("../data/Dataset 1/round 002/AM1c-s11-r002_A01_w2.tif")

# results, corrected = register.register_phasexcorr(I1, I2)
# # overlay = display.merge_images(I1, corrected)

# # plt.imshow(overlay)
# # plt.show()

# src, dst = register.calculate_displacement_field(
#     I1, corrected, search_window=200, grid_size=(20, 20)
# )

# print(len(src))
# print(len(dst))

# grid_size = (200, 200)

# tform, full_src, full_dst = register.estimate_tform(
#     src, dst, I1.shape, mesh_grid=grid_size
# )

# print("Warping")
# # corrected = sk.transform.warp(I2, tform, output_shape=I1.shape)
# corrected2 = register.fast_warp(
#     corrected, I1.shape, full_src, full_dst, mesh_grid=grid_size
# )

# print("Done")

# overlay = display.merge_images(I1, corrected2)

# fig = plt.figure(figsize=(8, 3))
# ax1 = plt.subplot(1, 2, 1)
# ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)

# ax1.imshow(overlay)
# ax1.set_axis_off()
# ax1.title("Fine registration")

# orov = display.merge_images(I1, corrected)
# ax2.imshow(orov)
# ax2.set_axis_off()
# ax2.title("Coarse registration")

# plt.show()

# # register.generate_quiver_plot(I1, full_src, full_dst)

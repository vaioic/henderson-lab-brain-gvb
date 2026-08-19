# Load the cellpose mask
from pathlib import Path

import numpy as np
import oic_toolkit
import skimage
from matplotlib import pyplot as plt

output_dir = Path(r"../processed/2026-08-19 Dev")
output_dir.mkdir(exist_ok=True, parents=True)

cell_mask = skimage.io.imread("../processed/2026-08-17 Dev/cell_masks.tif")

# Read the green channel
psyn_image = skimage.io.imread(
    r"../processed/20260814_registered_images/AW GVB AM1c-s11 010426_Plate_4536_registered/AW GVB AM1c-s11 010426_A01_channel0.tif"
)

dapi_image = skimage.io.imread(
    r"../test/warped_dataset_output_4555_2/warped_AM1c-s11-r002_A01_channel1.tif"
)

image_list = [
    # r"..\processed\20260814_registered_images\AW GVB AM1c-s11 010426_Plate_4536_registered\AW GVB AM1c-s11 010426_A01_channel0.tif",
    r"..\processed\20260814_registered_images\AW GVB AM1c-s11 010426_Plate_4536_registered\AW GVB AM1c-s11 010426_A01_channel2.tif",
    # r"..\processed\20260814_registered_images\AW GVB AM1c-s11 010426_Plate_4536_registered\AW GVB AM1c-s11 010426_A01_channel3.tif",
    # r"../test/warped_dataset_output_4555_2/warped_AM1c-s11-r002_A01_channel0.tif",
    r"../test/warped_dataset_output_4555_2/warped_AM1c-s11-r002_A01_channel2.tif",
    r"../test/warped_dataset_output_4555_2/warped_AM1c-s11-r002_A01_channel3.tif",
]

# Measure the pSyn signal
cell_props_psyn = skimage.measure.regionprops_table(
    cell_mask, psyn_image, properties=("intensity_mean", "label", "coords", "bbox")
)

# Plot the intensity histogram
# plt.hist(cell_props_psyn["intensity_mean"], bins=150)
# plt.show()
# exit()

# Process positive cells
final_cell_mask = np.zeros_like(cell_mask)

for idx in range(len(cell_props_psyn["label"])):
    # TODO: Load LAMP1? To determine if it's a real cell

    if cell_props_psyn["intensity_mean"][idx] >= 450:
        coords = cell_props_psyn["coords"][idx]

        final_cell_mask[coords[:, 0], coords[:, 1]] = idx

    # # Find spots in the cell
    # min_row, min_col, max_row, max_col = cell_props_psyn["bbox"]

    # for idx, file in enumerate(image_list):
    #     image = skimage.io.imread(file)

    #     # # Background subtract
    #     # image_bgsub = skimage.morphology.white_tophat(image, bg_footprint)

    #     # image_bgsub_crop = image_bgsub[ROI[1] : ROI[3], ROI[0] : ROI[2]]

    #     # plt.imshow(image_bgsub_crop)
    #     # plt.show()

    #     diff_img = segment.difference_of_gaussians(image, d_min=3, d_max=15)

    #     # Calculate a threshold for the spots
    #     diff_median = np.median(diff_img)
    #     diff_std = np.std(diff_img)

    #     spot_thresh = diff_median + (5 * diff_std)

    #     mask_spot = diff_img > spot_thresh

    #     labels = skimage.measure.label(mask_spot)
    #     props = skimage.measure.regionprops_table(
    #         labels, image, properties=("centroid", "max_intensity")
    #     )

    #     all_props.append(props)

    #     df = pd.DataFrame(props)
    #     df = filter_close_regions_from_table(df, 10)
    #     df["marker"] = marker_name[idx]
    #     all_df.append(df)

    #     # --- PLOTTING WITH 'X' MARKERS ---
    #     plt.figure(figsize=(8, 8))

    #     # Convert image
    #     p_low, p_high = np.percentile(image, (45, 98))

    #     # values, counts = np.unique(image, return_counts=True)

    #     # # Find the index of the highest count
    #     # mode_index = np.argmax(counts)
    #     # mode_value = values[mode_index]

    #     # print(mode_value)

    #     image_norm = skimage.exposure.rescale_intensity(
    #         image, in_range=(p_low, p_high), out_range=(0.0, 1.0)
    #     )

    #     image_rgb = skimage.color.gray2rgb(image_norm)

    #     for p in range(len(df["centroid-0"])):
    #         center_r = int(df["centroid-0"][p])
    #         center_c = int(df["centroid-1"][p])
    #         circle_radius = 3
    #         rr, cc = circle_perimeter(
    #             center_r, center_c, circle_radius, shape=image_rgb.shape
    #         )
    #         image_rgb[rr, cc] = [1, 0, 0]  # Set to Red [R, G, B]

    #     skimage.io.imsave(
    #         output_dir / ("spots_" + marker_name[idx] + ".png"),
    #         skimage.util.img_as_ubyte(image_rgb),
    #     )


# BBOX: (min_row, min_col, max_row, max_col)
p_low, p_high = np.percentile(psyn_image, (45, 98))
psyn_image_out = skimage.exposure.rescale_intensity(
    psyn_image, in_range=(p_low, p_high), out_range=(0.0, 1.0)
)
# psyn_image_out = skimage.util.img_as_ubyte(psyn_image_out)

final_cell_mask_out = skimage.color.label2rgb(final_cell_mask)

positive_cell_overlay = oic_toolkit.display.merge_images(
    psyn_image_out, final_cell_mask_out
)
skimage.io.imsave(output_dir / "positive_cell_overlay.png", positive_cell_overlay)

exit()

plt.imshow(psyn_image)
plt.show()


# Look for spots

# Cluster to get GVBs

# Export data to CSV

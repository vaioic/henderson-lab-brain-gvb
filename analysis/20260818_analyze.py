# Load the cellpose mask
from pathlib import Path

import numpy as np
import oic_toolkit
import pandas as pd
import skimage
import tifffile as tiff
from scipy.spatial import cKDTree

gvb_size = 6  # (Approx. 1.04 micron)

output_dir = Path(r"../processed/2026-09-18 Dev")
output_dir.mkdir(exist_ok=True, parents=True)

cell_mask = skimage.io.imread("../processed/2026-08-17 Dev/cell_masks.tif")

# Read the green channel
psyn_image = skimage.io.imread(
    r"../processed/20260814_registered_images/AW GVB AM1c-s11 010426_Plate_4536_registered/AW GVB AM1c-s11 010426_A01_channel0.tif"
)

dapi_image = skimage.io.imread(
    r"../test/warped_dataset_output_4555_2/warped_AM1c-s11-r002_A01_channel1.tif"
)

nuclear_mask = skimage.io.imread(r"../processed/2026-08-26 Dev/cell_masks.tif")

# Expand the nuclear mask to get the cell masks
cell_mask = skimage.segmentation.expand_labels(nuclear_mask, distance=12.5)

# overlay = oic_toolkit.display.merge_images(dapi_image, cell_mask)
# plt.imshow(overlay)
# plt.show()


image_list = [
    # r"..\processed\20260814_registered_images\AW GVB AM1c-s11 010426_Plate_4536_registered\AW GVB AM1c-s11 010426_A01_channel0.tif",
    r"..\processed\20260814_registered_images\AW GVB AM1c-s11 010426_Plate_4536_registered\AW GVB AM1c-s11 010426_A01_channel2.tif",
    r"..\processed\20260814_registered_images\AW GVB AM1c-s11 010426_Plate_4536_registered\AW GVB AM1c-s11 010426_A01_channel3.tif",
    # r"../test/warped_dataset_output_4555_2/warped_AM1c-s11-r002_A01_channel0.tif",
    r"../test/warped_dataset_output_4555_2/warped_AM1c-s11-r002_A01_channel2.tif",
    r"../test/warped_dataset_output_4555_2/warped_AM1c-s11-r002_A01_channel3.tif",
]

marker_name = [
    # "pSyn",
    "pTau",
    "CHMP2B",
    # "LAMP1",
    "pMARK",
    "CK1delta",
]


# Plot the intensity histogram
# plt.hist(cell_props_psyn["intensity_mean"], bins=150)
# plt.show()
# exit()

# Process positive cells
# Measure the pSyn signal
cell_props_psyn = skimage.measure.regionprops_table(
    cell_mask, psyn_image, properties=("intensity_mean", "label", "coords", "bbox")
)

final_cell_mask = np.zeros_like(cell_mask)

for idx in range(len(cell_props_psyn["label"])):
    # TODO: Load LAMP1? To determine if it's a real cell

    if cell_props_psyn["intensity_mean"][idx] >= 450:
        coords = cell_props_psyn["coords"][idx]

        final_cell_mask[coords[:, 0], coords[:, 1]] = idx


H0, W0 = cell_mask.shape[:2]

# Run the spot segmentation algorithm on all the images
all_spot_df = []
for idx, file in enumerate(image_list):
    image = skimage.io.imread(file)

    diff_img = oic_toolkit.segment.difference_of_gaussians(image, d_min=3, d_max=15)

    # Calculate a threshold for the spots
    diff_median = np.median(diff_img)
    diff_std = np.std(diff_img)

    spot_thresh = diff_median + (5 * diff_std)

    mask_spot = diff_img > spot_thresh

    # Filter out spots that are not in a positive cell
    final_mask_spot = mask_spot.copy()
    final_mask_spot[final_cell_mask == 0] = False

    curr_spot_label = skimage.measure.label(final_mask_spot)

    # Get position of spots
    spot_props = skimage.measure.regionprops_table(
        curr_spot_label,
        final_cell_mask,
        properties=("label", "centroid", "feret_diameter_max"),
    )
    # TODO: In future, maybe replace feret diameter with a fitted profile

    # Measure intensity of the spots
    spot_props_intensity = skimage.measure.regionprops_table(
        curr_spot_label,
        image,
        properties=("intensity_mean",),
    )

    # Determine cell ID for each spot
    cell_id = []
    for spot_idx in range(len(spot_props["label"])):
        spot_y = int(spot_props["centroid-0"][spot_idx])
        spot_x = int(spot_props["centroid-1"][spot_idx])

        cell_id.append(final_cell_mask[spot_y, spot_x])

    spot_props["cell_id"] = np.array(cell_id)

    spot_props["intensity"] = spot_props_intensity["intensity_mean"]

    # Convert to DataFrame and store channel name
    spot_df = pd.DataFrame(spot_props)
    spot_df["channel"] = marker_name[idx]
    spot_df["feret_diameter_max"] = spot_df["feret_diameter_max"] * 0.1726

    all_spot_df.append(spot_df)

    # Save the unfiltered spot mask
    skimage.io.imsave(
        output_dir / ("raw_spots_mask_" + marker_name[idx] + ".tif"),
        final_mask_spot,
    )

df_all_spots = pd.concat(all_spot_df, ignore_index=True)

# --- Try to identify the GVBs by clustering --- #

from sklearn.cluster import MeanShift, estimate_bandwidth

coords = df_all_spots[["centroid-0", "centroid-1"]].to_numpy()

bandwidth = estimate_bandwidth(coords, quantile=0.2, n_samples=500)


ms = MeanShift(bandwidth=gvb_size, bin_seeding=True)
ms.fit(coords)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

df_all_spots["cluster_id"] = labels

df_all_spots.to_csv(output_dir / "spot_data.csv", index=False)

## -- Visualization --- ##

# Make a plot by adding the circles from the spots we identified, over an image of the
# nuclei?


# ## Visualized the GVBs by plotting the identified cluster over the image?
# img1 = skimage.io.imread(image_list[0])

# p_low, p_high = np.percentile(img1, (45, 98))
# img1 = skimage.exposure.rescale_intensity(
#     img1, in_range=(p_low, p_high), out_range=(0.0, 0.6)
# )


# img2 = skimage.io.imread(image_list[1])
# img2 = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))

# p_low, p_high = np.percentile(img2, (45, 98))
# img2 = skimage.exposure.rescale_intensity(
#     img2, in_range=(p_low, p_high), out_range=(0.0, 0.6)
# )

# magenta = np.array([1.0, 0.0, 1.0])  # Channel 1
# cyan = np.array([0.0, 1.0, 1.0])  # Channel 2
# yellow = np.array([1.0, 1.0, 0.0])  # Channel 3

# composite = (
#     img1[..., None] * magenta + img2[..., None] * cyan + img1[..., None] * yellow
# )
# composite = np.clip(composite, 0.0, 1.0)

# from skimage.draw import circle_perimeter

# Make a picture showing the GVB identification
image_with_circles = np.zeros_like(dapi_image)

# image_with_circles = composite.copy()
H, W = image_with_circles.shape[:2]

radius = int(gvb_size)
white_color = np.array([1.0, 1.0, 0.0])  # RGB white

# MeanShift centers output as (row, col) -> (y, x)
for cy, cx in cluster_centers:
    r_center, c_center = int(round(cy)), int(round(cx))

    # Get perimeter coordinates, bounded by image dimensions
    rr, cc = skimage.draw.circle_perimeter(r_center, c_center, radius, shape=(H, W))

    # Calculate angles for each perimeter pixel to create a dotted effect
    angles = np.arctan2(rr - r_center, cc - c_center)
    # Filter pixels to create 12 distinct dots around the circle
    dotted_mask = np.sin(12 * angles) > 0

    # Draw white pixels onto the array
    image_with_circles[rr[dotted_mask], cc[dotted_mask]] = 1.0

image_with_circles = (image_with_circles * 65535).astype(np.uint16)

# Draw all the spots
# skimage.io.imsave(
#     output_dir / "identified_GVB.png", skimage.util.img_as_ubyte(image_with_circles)
# )

spots_image = np.zeros_like(dapi_image)

for x, y in coords:
    rr, cc = skimage.draw.disk(
        (
            x,
            y,
        ),
        3,
        shape=spots_image.shape,
    )
    spots_image[rr, cc] = 1.0

spots_image = (spots_image * 65535).astype(np.uint16)

# # Draw all the images into a giant merged image
image_list_2 = [
    r"..\processed\20260814_registered_images\AW GVB AM1c-s11 010426_Plate_4536_registered\AW GVB AM1c-s11 010426_A01_channel0.tif",
    r"..\processed\20260814_registered_images\AW GVB AM1c-s11 010426_Plate_4536_registered\AW GVB AM1c-s11 010426_A01_channel1.tif",
    r"..\processed\20260814_registered_images\AW GVB AM1c-s11 010426_Plate_4536_registered\AW GVB AM1c-s11 010426_A01_channel2.tif",
    r"..\processed\20260814_registered_images\AW GVB AM1c-s11 010426_Plate_4536_registered\AW GVB AM1c-s11 010426_A01_channel3.tif",
    r"../test/warped_dataset_output_4555_2/warped_AM1c-s11-r002_A01_channel0.tif",
    r"../test/warped_dataset_output_4555_2/warped_AM1c-s11-r002_A01_channel1.tif",
    r"../test/warped_dataset_output_4555_2/warped_AM1c-s11-r002_A01_channel2.tif",
    r"../test/warped_dataset_output_4555_2/warped_AM1c-s11-r002_A01_channel3.tif",
]

marker_name_2 = [
    "pSyn",
    "DAPI 2",
    "pTau",
    "CHMP2B",
    "LAMP1",
    "DAPI 1",
    "pMARK",
    "CK1delta",
]


images = [skimage.io.imread(f) for f in image_list_2]
images.append(final_cell_mask)
images.append(image_with_circles)
images.append(spots_image)

marker_name_2.append("Cell mask")
marker_name_2.append("GVB")
marker_name_2.append("Spots")

stack = np.stack(images, axis=0)

tile_size = 256
subresolutions = 2

with tiff.TiffWriter(output_dir / "combined_2.ome.tif", bigtiff=True) as tif:
    options = {
        "photometric": "minisblack",
        "tile": (tile_size, tile_size),
        "compression": "lzw",
        # "resolution": (1e4 / 0.1726, 1e4 / 0.1726),
        "resolutionunit": "CENTIMETER",
    }

    tif.write(
        stack,
        subifds=subresolutions,
        metadata={
            "axes": "CYX",
            "Channel": {"Name": marker_name_2},
            # "PhysicalSizeX": 0.1726,
            # "PhysicalSizeXUnit": "µm",
            # "PhysicalSizeY": 0.1726,
            # "PhysicalSizeYUnit": "µm",
        },
        **options,
    )

    for level in range(subresolutions):
        mag = 2 ** (level + 1)
        tif.write(
            stack[:, ::mag, ::mag],
            subfiletype=1,  # FILETYPE.REDUCEDIMAGE
            **options,
        )


exit()


def filter_close_regions_from_table(
    props_table, min_distance, intensity_key="max_intensity"
):
    """
    Filters a regionprops_table dictionary to remove close centroids,
    keeping the ones with higher intensity.

    Parameters:
    - props_table: dict returned by skimage.measure.regionprops_table
    - min_distance: minimum allowed distance between centroids
    - intensity_key: column used to prioritize which region to keep (e.g., 'max_intensity' or 'mean_intensity')

    Returns:
    - filtered_table: a Pandas DataFrame containing only the filtered rows
    """
    # Convert dictionary to DataFrame for easier row filtering
    df = pd.DataFrame(props_table)

    if len(df) == 0:
        return df

    # Extract coordinate columns (handles 2D: centroid-0, centroid-1)
    coord_cols = [col for col in df.columns if col.startswith("centroid-")]
    centroids = df[coord_cols].to_numpy()

    # Check if intensity sorting key exists, otherwise fallback to arbitrary order
    if intensity_key in df.columns:
        intensities = df[intensity_key].to_numpy()
        # Sort descending (brightest first)
        sorted_indices = np.argsort(intensities)[::-1]
    else:
        sorted_indices = np.arange(len(df))

    centroids_sorted = centroids[sorted_indices]

    # Build KD-Tree and find pairs within min_distance
    tree = cKDTree(centroids_sorted)
    pairs = tree.query_pairs(min_distance)

    # Greedily discard the lower-priority (later in sorted list) neighbor
    to_remove_sorted_idx = set()
    for i, j in pairs:
        if i not in to_remove_sorted_idx and j not in to_remove_sorted_idx:
            to_remove_sorted_idx.add(j)

    # Map back to original DataFrame indices
    to_remove_original_idx = [sorted_indices[j] for j in to_remove_sorted_idx]

    # Drop rows that are too close
    filtered_df = df.drop(index=to_remove_original_idx).reset_index(drop=True)

    return filtered_df


bg_footprint = skimage.morphology.disk(20)
all_props = []
all_df = []


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

# plt.imshow(psyn_image)
# plt.show()


# Merge all the images and spots together

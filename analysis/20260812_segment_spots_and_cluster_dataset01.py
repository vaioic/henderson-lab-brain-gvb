# Import an ROI and look for spots
from pathlib import Path

import numpy as np
import pandas as pd
import skimage
from matplotlib import pyplot as plt
from oic_toolkit import segment
from skimage.draw import circle_perimeter

# image_list = [
#     r"../processed/20260815_reregistered_moving_images/warped_dataset_output_4555/warped_AM1c-s11-r002_A01_channel0.tif",
#     r"../processed/20260815_reregistered_moving_images/warped_dataset_output_4555/warped_AM1c-s11-r002_A01_channel2.tif",
#     r"../processed/20260815_reregistered_moving_images/warped_dataset_output_4555/warped_AM1c-s11-r002_A01_channel3.tif",
#     r"../processed/20260814_registered_images/AW GVB AM1c-s11 010426_Plate_4536_registered/AW GVB AM1c-s11 010426_A01_channel0.tif",
#     r"../processed/20260814_registered_images/AW GVB AM1c-s11 010426_Plate_4536_registered/AW GVB AM1c-s11 010426_A01_channel3.tif",
#     r"../processed/20260814_registered_images/AW GVB AM1c-s11 010426_Plate_4536_registered/AW GVB AM1c-s11 010426_A01_channel4.tif",
# ]

image_list = [
    # r"..\processed\20260814_registered_images\AW GVB AM1c-s11 010426_Plate_4536_registered\AW GVB AM1c-s11 010426_A01_channel0.tif",
    r"..\processed\20260814_registered_images\AW GVB AM1c-s11 010426_Plate_4536_registered\AW GVB AM1c-s11 010426_A01_channel2.tif",
    # r"..\processed\20260814_registered_images\AW GVB AM1c-s11 010426_Plate_4536_registered\AW GVB AM1c-s11 010426_A01_channel3.tif",
    # r"../test/warped_dataset_output_4555_2/warped_AM1c-s11-r002_A01_channel0.tif",
    r"../test/warped_dataset_output_4555_2/warped_AM1c-s11-r002_A01_channel2.tif",
    r"../test/warped_dataset_output_4555_2/warped_AM1c-s11-r002_A01_channel3.tif",
]

output_dir = Path(r"../processed/2026-08-17 Dev")
output_dir.mkdir(exist_ok=True, parents=True)

marker_name = [
    # "LAMP1",
    "pMARK",
    # "CK1delta",
    # "pSyn",
    "pTau",
    "CHMP2B",
]

all_props = []
all_df = []

from scipy.spatial import cKDTree


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
ROI = [8567, 1195, 8818, 1446]
for idx, file in enumerate(image_list):
    image = skimage.io.imread(file)

    # # Background subtract
    # image_bgsub = skimage.morphology.white_tophat(image, bg_footprint)

    # image_bgsub_crop = image_bgsub[ROI[1] : ROI[3], ROI[0] : ROI[2]]

    # plt.imshow(image_bgsub_crop)
    # plt.show()

    diff_img = segment.difference_of_gaussians(image, d_min=3, d_max=15)

    # Calculate a threshold for the spots
    diff_median = np.median(diff_img)
    diff_std = np.std(diff_img)

    spot_thresh = diff_median + (5 * diff_std)

    mask_spot = diff_img > spot_thresh

    labels = skimage.measure.label(mask_spot)
    props = skimage.measure.regionprops_table(
        labels, image, properties=("centroid", "max_intensity")
    )

    all_props.append(props)

    df = pd.DataFrame(props)
    df = filter_close_regions_from_table(df, 10)
    df["marker"] = marker_name[idx]
    all_df.append(df)

    # --- PLOTTING WITH 'X' MARKERS ---
    plt.figure(figsize=(8, 8))

    # Convert image
    p_low, p_high = np.percentile(image, (45, 98))

    # values, counts = np.unique(image, return_counts=True)

    # # Find the index of the highest count
    # mode_index = np.argmax(counts)
    # mode_value = values[mode_index]

    # print(mode_value)

    image_norm = skimage.exposure.rescale_intensity(
        image, in_range=(p_low, p_high), out_range=(0.0, 1.0)
    )

    image_rgb = skimage.color.gray2rgb(image_norm)

    for p in range(len(df["centroid-0"])):
        center_r = int(df["centroid-0"][p])
        center_c = int(df["centroid-1"][p])
        circle_radius = 3
        rr, cc = circle_perimeter(
            center_r, center_c, circle_radius, shape=image_rgb.shape
        )
        image_rgb[rr, cc] = [1, 0, 0]  # Set to Red [R, G, B]

    skimage.io.imsave(
        output_dir / ("spots_" + marker_name[idx] + ".png"),
        skimage.util.img_as_ubyte(image_rgb),
    )

    # plt.imshow(image_norm, cmap="gray")

    # # regionprops returns centroids as (row, col) which map to (y, x)
    # # df['centroid-0'] is the Y coordinate (row)
    # # df['centroid-1'] is the X coordinate (column)
    # plt.plot(
    #     df["centroid-1"], df["centroid-0"], "rx", markersize=6, markeredgewidth=1.5
    # )

    # plt.title(f"Centroids for {marker_name[idx]}")
    # plt.axis("off")
    # plt.show()
    # exit()

    # Save ths spot mask

    # Save some kind of overlay showing detected spots vs image

combined_df = pd.concat(all_df, ignore_index=True)
combined_df.to_csv("spot_data.csv", index=False)

from sklearn.cluster import MeanShift, estimate_bandwidth

# Concatenate the centroids
for idx, p in enumerate(all_props):
    if idx == 0:
        coords = np.column_stack((p["centroid-0"], p["centroid-1"]))
    else:
        curr_coords = np.column_stack((p["centroid-0"], p["centroid-1"]))
        coords = np.vstack((coords, curr_coords))

bandwidth = estimate_bandwidth(coords, quantile=0.2, n_samples=500)

gvb_size = 30.0

ms = MeanShift(bandwidth=gvb_size, bin_seeding=True)
ms.fit(coords)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

## -- Visualization --- ##

# Make a plot by adding the circles from the spots we identified, over an image of the
# nuclei?


## Visualized the GVBs by plotting the identified cluster over the image?
img1 = skimage.io.imread(image_list[0])

p_low, p_high = np.percentile(img1, (45, 98))
img1 = skimage.exposure.rescale_intensity(
    img1, in_range=(p_low, p_high), out_range=(0.0, 0.6)
)


img2 = skimage.io.imread(image_list[1])
img2 = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))

p_low, p_high = np.percentile(img2, (45, 98))
img2 = skimage.exposure.rescale_intensity(
    img2, in_range=(p_low, p_high), out_range=(0.0, 0.6)
)

magenta = np.array([1.0, 0.0, 1.0])  # Channel 1
cyan = np.array([0.0, 1.0, 1.0])  # Channel 2
yellow = np.array([1.0, 1.0, 0.0])  # Channel 3

composite = (
    img1[..., None] * magenta + img2[..., None] * cyan + img1[..., None] * yellow
)
composite = np.clip(composite, 0.0, 1.0)


from skimage.draw import circle_perimeter

# Create a copy of composite to modify directly
image_with_circles = composite.copy()
H, W, _ = image_with_circles.shape

radius = int(gvb_size)
white_color = np.array([1.0, 1.0, 0.0])  # RGB white

# MeanShift centers output as (row, col) -> (y, x)
for cy, cx in cluster_centers:
    r_center, c_center = int(round(cy)), int(round(cx))

    # Get perimeter coordinates, bounded by image dimensions
    rr, cc = circle_perimeter(r_center, c_center, radius, shape=(H, W))

    # Calculate angles for each perimeter pixel to create a dotted effect
    angles = np.arctan2(rr - r_center, cc - c_center)
    # Filter pixels to create 12 distinct dots around the circle
    dotted_mask = np.sin(12 * angles) > 0

    # Draw white pixels onto the array
    image_with_circles[rr[dotted_mask], cc[dotted_mask]] = white_color

skimage.io.imsave(
    output_dir / "identified_GVB.png", skimage.util.img_as_ubyte(image_with_circles)
)

# Export GVB data


exit()

# plt.imshow(composite)
# plt.show()

# plt.close()
# exit()


# centers_xy = cluster_centers[:, [1, 0]]

# fig, ax = plt.subplots(figsize=(24, 24), dpi=150)
# plt.imshow(composite)

# # Plot cluster centers
# # ax.scatter(centers_xy[:, 0], centers_xy[:, 1], c="white", s=6, zorder=3)

# # Vectorized 20 px dotted circles
# patches = [mpatches.Circle((x, y), radius=20, fill=False) for x, y in centers_xy]
# collection = PatchCollection(
#     patches,
#     match_original=True,
#     facecolors="none",
#     edgecolors="white",
#     linestyles=":",
#     linewidths=1,
# )
# ax.add_collection(collection)

# plt.axis("off")
# # plt.show()
# plt.savefig("spot_clusters.png", dpi=300, bbox_inches="tight")

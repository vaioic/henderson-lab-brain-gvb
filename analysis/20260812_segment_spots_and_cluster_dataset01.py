# Import an ROI and look for spots
import pandas as pd
import skimage
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from oic_toolkit import segment

image_list = [
    r"../processed/shading_corrected/AM1c-s11-r002_Plate_4555_shifted/AM1c-s11-r002_A01_channel1_channel1.tif",
    r"../processed/shading_corrected/AM1c-s11-r002_Plate_4555_shifted/AM1c-s11-r002_A01_channel3_channel3.tif",
    r"../processed/shading_corrected/AM1c-s11-r002_Plate_4555_shifted/AM1c-s11-r002_A01_channel4_channel4.tif",
    r"../processed/shading_corrected/AW GVB AM1c-s11 010426_Plate_4536_shifted2/AW GVB AM1c-s11 010426_A01_channel1_channel1.tif",
    r"../processed/shading_corrected/AW GVB AM1c-s11 010426_Plate_4536_shifted2/AW GVB AM1c-s11 010426_A01_channel3_channel3.tif",
    r"../processed/shading_corrected/AW GVB AM1c-s11 010426_Plate_4536_shifted2/AW GVB AM1c-s11 010426_A01_channel4_channel4.tif",
]

marker_name = [
    "plate_4555_ch1",
    "plate_4555_ch3",
    "plate_4555_ch4",
    "plate_4536_ch1",
    "plate_4536_ch3",
    "plate_4536_ch4",
]

all_props = []
all_df = []

for idx, file in enumerate(image_list):
    image = skimage.io.imread(file)

    diff_img = segment.difference_of_gaussians(image, d_min=3, d_max=15)
    mask_spot = diff_img > 0.2

    labels = skimage.measure.label(mask_spot)
    props = skimage.measure.regionprops_table(labels, properties=("centroid",))

    all_props.append(props)

    df = pd.DataFrame(props)
    df["marker"] = marker_name[idx]
    all_df.append(df)

combined_df = pd.concat(all_df, ignore_index=True)
combined_df.to_csv("spot_data.csv", index=False)

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

# Concatenate the centroids
for idx, p in enumerate(all_props):
    if idx == 0:
        coords = np.column_stack((p["centroid-0"], p["centroid-1"]))
    else:
        curr_coords = np.column_stack((p["centroid-0"], p["centroid-1"]))
        coords = np.vstack((coords, curr_coords))

bandwidth = estimate_bandwidth(coords, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=20.0, bin_seeding=True)
ms.fit(coords)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

# fig, ax = plt.subplots(figsize=(8, 8))

# # Optional: Show the underlying mask in the background for spatial context
# # ax.imshow(mask_spot, cmap='gray', alpha=0.5)

# # Separate noise points (-1) from actual clusters if using DBSCAN
# unique_labels = set(labels)

# # Use a colormap for the clusters
# colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black/Gray color and different marker for background noise/outliers
#         col = [0.5, 0.5, 0.5, 0.5]
#         marker = "x"
#         label_name = "Noise / Single spots"
#     else:
#         marker = "o"
#         label_name = f"Cluster {k}"

#     # Extract coordinates for this specific label
#     class_member_mask = labels == k
#     xy = coords[class_member_mask]

#     # Note: matplotlib scatter takes (x, y), but skimage centroids are (y, x)
#     ax.scatter(
#         xy[:, 1],
#         xy[:, 0],
#         c=[col],
#         marker=marker,
#         s=50,
#         label=label_name,
#         edgecolors="k",
#         linewidths=0.5,
#     )

# ax.set_title(
#     f"Spot Clustering Results (Total groups: {len(unique_labels) - (1 if -1 in unique_labels else 0)})"
# )
# ax.set_xlabel("X Pixel")
# ax.set_ylabel("Y Pixel")
# ax.invert_yaxis()  # Match image coordinate conventions (origin at top-left)
# ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")

# plt.tight_layout()
# plt.show()

## Visualized the GVBs by plotting the identified cluster over the image?
img1 = skimage.io.imread(image_list[3])
img1 = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1))

img2 = skimage.io.imread(image_list[4])
img2 = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))

img3 = skimage.io.imread(image_list[2])
img3 = (img3 - np.min(img3)) / (np.max(img3) - np.min(img3))

magenta = np.array([1.0, 0.0, 1.0])  # Channel 1
cyan = np.array([0.0, 1.0, 1.0])  # Channel 2
yellow = np.array([1.0, 1.0, 0.0])  # Channel 3

composite = (
    img1[..., None] * magenta + img2[..., None] * cyan + img3[..., None] * yellow
)
composite = np.clip(composite, 0.0, 1.0)


from skimage.draw import circle_perimeter

# Create a copy of composite to modify directly
image_with_circles = composite.copy()
H, W, _ = image_with_circles.shape

radius = 20
white_color = np.array([1.0, 1.0, 1.0])  # RGB white

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

skimage.io.imsave("spots_out.png", skimage.util.img_as_ubyte(image_with_circles))
exit()

# plt.imshow(composite)
# plt.show()

# plt.close()
# exit()


centers_xy = cluster_centers[:, [1, 0]]

fig, ax = plt.subplots(figsize=(24, 24), dpi=150)
plt.imshow(composite)

# Plot cluster centers
# ax.scatter(centers_xy[:, 0], centers_xy[:, 1], c="white", s=6, zorder=3)

# Vectorized 20 px dotted circles
patches = [mpatches.Circle((x, y), radius=20, fill=False) for x, y in centers_xy]
collection = PatchCollection(
    patches,
    match_original=True,
    facecolors="none",
    edgecolors="white",
    linestyles=":",
    linewidths=1,
)
ax.add_collection(collection)

plt.axis("off")
# plt.show()
plt.savefig("spot_clusters.png", dpi=300, bbox_inches="tight")

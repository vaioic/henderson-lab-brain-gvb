# Import an ROI and look for spots
import skimage
from matplotlib import pyplot as plt
from oic_toolkit import segment

ROI = [6896, 154, 10963, 3498]  # (x, y, x, y)

img1 = skimage.io.imread(
    "../processed/shading_corrected/AM1c-s11-r002_Plate_4555_shifted/AM1c-s11-r002_A01_channel4_channel4.tif"
)

img1_crop = img1[ROI[1] : ROI[3], ROI[0] : ROI[2]]

# plt.imshow(img1_crop)
# plt.show()

diff_img = segment.difference_of_gaussians(img1, d_min=3, d_max=15)

mask_spot = diff_img > 0.2

# plt.imshow(mask_spot)
# plt.show()

img2 = skimage.io.imread(
    "../processed/shading_corrected/AM1c-s11-r002_Plate_4555_shifted/AM1c-s11-r002_A01_channel3_channel3.tif"
)

img2_crop = img2[ROI[1] : ROI[3], ROI[0] : ROI[2]]

diff_img2 = segment.difference_of_gaussians(img2, d_min=3, d_max=15)

mask_spot2 = diff_img2 > 0.2

# plt.imshow(mask_spot2)
# plt.show()

# Now try to cluster the spots
labels1 = skimage.measure.label(mask_spot)
props1 = skimage.measure.regionprops_table(labels1, properties=("centroid",))

labels2 = skimage.measure.label(mask_spot2)
props2 = skimage.measure.regionprops_table(labels2, properties=("centroid",))

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

# Concatenate the centroids
coords1 = np.column_stack((props1["centroid-0"], props1["centroid-1"]))
coords2 = np.column_stack((props2["centroid-0"], props2["centroid-1"]))

X = np.vstack((coords1, coords2))

bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=20.0, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

fig, ax = plt.subplots(figsize=(8, 8))

# Optional: Show the underlying mask in the background for spatial context
# ax.imshow(mask_spot, cmap='gray', alpha=0.5)

# Separate noise points (-1) from actual clusters if using DBSCAN
unique_labels = set(labels)

# Use a colormap for the clusters
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black/Gray color and different marker for background noise/outliers
        col = [0.5, 0.5, 0.5, 0.5]
        marker = "x"
        label_name = "Noise / Single spots"
    else:
        marker = "o"
        label_name = f"Cluster {k}"

    # Extract coordinates for this specific label
    class_member_mask = labels == k
    xy = X[class_member_mask]

    # Note: matplotlib scatter takes (x, y), but skimage centroids are (y, x)
    ax.scatter(
        xy[:, 1],
        xy[:, 0],
        c=[col],
        marker=marker,
        s=50,
        label=label_name,
        edgecolors="k",
        linewidths=0.5,
    )

ax.set_title(
    f"Spot Clustering Results (Total groups: {len(unique_labels) - (1 if -1 in unique_labels else 0)})"
)
ax.set_xlabel("X Pixel")
ax.set_ylabel("Y Pixel")
ax.invert_yaxis()  # Match image coordinate conventions (origin at top-left)
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")

plt.tight_layout()
plt.show()
plt.savefig("spot_clusters.png")

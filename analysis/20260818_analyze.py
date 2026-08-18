# Load the cellpose mask
import skimage
from matplotlib import pyplot as plt

cell_mask = skimage.io.imread("../processed/2026-08-17 Dev/cell_masks.tif")

# Read the green channel
psyn_image = skimage.io.imread(
    r"../test/warped_dataset_output_4555_2/warped_AM1c-s11-r002_A01_channel0.tif"
)

# Measure the pSyn signal
cell_props_psyn = skimage.measure.regionprops_table(
    cell_mask, psyn_image, properties=("intensity_mean", "label", "coords")
)

final_cell_mask = cell_mask.copy()

for idx in range(len(cell_props_psyn["label"])):
    if cell_props_psyn["intensity_mean"][idx] <= 500:
        coords = cell_props_psyn["coords"][idx]

        final_cell_mask[coords[:, 0], coords[:, 1]] = False

plt.imshow(final_cell_mask)
plt.show()

exit()

plt.imshow(psyn_image)
plt.show()


# Look for spots

# Cluster to get GVBs

# Export data to CSV

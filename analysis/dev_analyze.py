# Import an ROI and look for spots

import oic_toolkit
import skimage
from matplotlib import pyplot as plt

img_1 = skimage.io.imread(
    r"../processed/shading_corrected/AM1c-s11-r002_Plate_4555/AM1c-s11-r002_A01_channel1.ome.tif"
)

img_3 = skimage.io.imread(
    r"../processed/shading_corrected/AM1c-s11-r002_Plate_4555/AM1c-s11-r002_A01_channel3.ome.tif"
)


img_4 = skimage.io.imread(
    r"../processed/shading_corrected/AM1c-s11-r002_Plate_4555/AM1c-s11-r002_A01_channel4.ome.tif"
)

ROI = [6200, 700, 7600, 2000]

img_1 = img_1[ROI[1] : ROI[3], ROI[0] : ROI[2]]
img_3 = img_3[ROI[1] : ROI[3], ROI[0] : ROI[2]]
img_4 = img_4[ROI[1] : ROI[3], ROI[0] : ROI[2]]

# Find spots
img_3_dog = oic_toolkit.segment.difference_of_gaussians(img_3, d_min=3, d_max=7)
img_3_spot_mask = img_3_dog > 0.2

img_4_dog = oic_toolkit.segment.difference_of_gaussians(img_4, d_min=3, d_max=7)
img_4_spot_mask = img_4_dog > 0.2

fig, axes = plt.subplots(3, 3, figsize=(10, 5))
axes[0, 0].imshow(img_1)
axes[0, 1].imshow(img_3)
axes[0, 2].imshow(img_4)

axes[1, 1].imshow(img_3_dog)
axes[1, 2].imshow(img_4_dog)

axes[2, 1].imshow(img_3_spot_mask)
axes[2, 2].imshow(img_4_spot_mask)
plt.show()


# Crop an ROI for testing

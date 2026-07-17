# Import an ROI and look for spots

from pathlib import Path

import oic_toolkit
import skimage
from matplotlib import pyplot as plt

data_dir = Path("../processed/shading_corrected/")

ROI = [5340, 172, 11950, 5870]

dataset1_dir = Path("AW GVB AM1c-s11 010426_Plate_4536")
dataset1_fn = "AW GVB AM1c-s11 010426_A01_channel"

dataset2_dir = Path("AM1c-s11-r002_Plate_4555")
dataset2_fn = "AM1c-s11-r002_A01_channel"

# Register the images
img1_dapi = skimage.io.imread(data_dir / dataset1_dir / (dataset1_fn + "2.tif"))
img1_dapi = img1_dapi[ROI[1] : ROI[3], ROI[0] : ROI[2]]

img2_dapi = skimage.io.imread(data_dir / dataset2_dir / (dataset2_fn + "2.tif"))
img2_dapi = img2_dapi[ROI[1] : ROI[3], ROI[0] : ROI[2]]

shift, img2_dapi_corr = oic_toolkit.register.phasexcorr(img1_dapi, img2_dapi)

merge = oic_toolkit.display.merge_images(img1_dapi, img2_dapi_corr)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img1_dapi)
axes[1].imshow(img2_dapi)
axes[2].imshow(merge)

plt.show()

exit()

# img_1 = skimage.io.imread(
#     r"../processed/shading_corrected/AW GVB AM1c-s11 010426_Plate_4536/AW GVB AM1c-s11 010426_A01_channel1.tif"
# )

# img_3 = skimage.io.imread(
#     r"../processed/shading_corrected/AW GVB AM1c-s11 010426_Plate_4536/AW GVB AM1c-s11 010426_A01_channel3.tif"
# )


# img_4 = skimage.io.imread(
#     r"../processed/shading_corrected/AW GVB AM1c-s11 010426_Plate_4536/AW GVB AM1c-s11 010426_A01_channel4.tif"
# )

# ROI = [6200, 700, 7600, 2000]
ROI = [8712, 2287, 10000, 3300]

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

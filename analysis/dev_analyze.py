# Import an ROI and look for spots

from pathlib import Path

import oic_toolkit
import skimage
from matplotlib import pyplot as plt

data_dir = Path("../processed/shading_corrected/")

ROI = [5340, 172, 11950, 5870]

dataset1_dir = Path("AW GVB AM1c-s11 010426_Plate_4536")
dataset1_fn = "AW GVB AM1c-s11 010426_A01_channel"

dataset2_dir = Path("AM1c-s11-r002_Plate_4555_shifted")
dataset2_fn = "AM1c-s11-r002_A01_channel"

# Find spots?
iC = 0

curr_img = skimage.io.imread(data_dir / dataset1_dir / (dataset1_fn + f"{iC + 1}.tif"))
curr_dog = oic_toolkit.segment.difference_of_gaussians(curr_img)
curr_spot_mask = curr_dog > 0.1

# ov = skimage.segmentation.mark_boundaries(curr_img, curr_spot_mask)

plt.imshow(curr_img)
plt.show()

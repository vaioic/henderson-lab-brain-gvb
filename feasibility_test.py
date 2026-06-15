import numpy as np
import skimage as sk
from matplotlib import pyplot as plt
from oic_toolkit import display, register, segment

from core import core_funcs

# Register the images
# TOP LEFT HEIGHT WIDTH

#ROI = [3842, 5627, 2200, 2200]
ROI = [8866, 1979, 3000, 3000]

I1 = sk.io.imread("../data/Dataset 1/round 001/AW GVB AM1c-s11 010426_A01_w2.tif")
I1 = I1[ROI[1]:(ROI[1] + ROI[3]), ROI[0]:(ROI[0] + ROI[2])]

ROI2 = [8422, 717, 3000, 3000]

I2 = sk.io.imread("../data/Dataset 1/round 002/AM1c-s11-r002_A01_w2.tif")
I2 = I2[ROI2[1]:(ROI2[1] + ROI2[3]), ROI2[0]:(ROI2[0] + ROI2[2])]

merge_originals = display.merge_images(I1, I2)

results, corrected_I2 = register.phasexcorr(I1, I2)


# # Try using optical flow
# u, v = register.optical_flow_tvl1(I1, corrected_I2)

# corrected_I2_flow = register.correct_optical_flow(corrected_I2, u, v)

# merge = display.merge_images(I1, corrected_I2_flow)

# plt.imshow(merge)
# plt.show()

# exit()


# merge_corrected = display.merge_images(I1, corrected_I2)

# Crop the image to leave only the registered part
crop_ROI = [368, 305, 1500, 2500] # Determined manually because the tiling is messing up the registration

I1_cropped = I1[crop_ROI[1]:(crop_ROI[1] + crop_ROI[3]), crop_ROI[0]:(crop_ROI[0] + crop_ROI[2])]
correctedI2_cropped = corrected_I2[crop_ROI[1]:(crop_ROI[1] + crop_ROI[3]), crop_ROI[0]:(crop_ROI[0] + crop_ROI[2])]

merge_corrected = display.merge_images(I1, corrected_I2)

plt.subplot(1, 2, 1)
plt.imshow(merge_originals)
plt.title("Original (merged)")

plt.subplot(1, 2, 2)
plt.imshow(merge_corrected)
plt.title("Corrected (merged)")

plt.show()
plt.close()
# exit()

# TODO: Apply registration to the other channels

# Load the other channels
alpha_synuclein = sk.io.imread("../data/Dataset 1/round 001/AW GVB AM1c-s11 010426_A01_w1.tif")
alpha_synuclein = alpha_synuclein[ROI[1]:(ROI[1] + ROI[3]), ROI[0]:(ROI[0] + ROI[2])]
alpha_synuclein = alpha_synuclein[crop_ROI[1]:(crop_ROI[1] + crop_ROI[3]), crop_ROI[0]:(crop_ROI[0] + crop_ROI[2])]

# plt.imshow(sk.exposure.rescale_intensity(alpha_synuclein, out_range=(0.0, 1.0)))
# plt.show()
# exit()

pTau = sk.io.imread("../data/Dataset 1/round 001/AW GVB AM1c-s11 010426_A01_w3.tif")
pTau = pTau[ROI[1]:(ROI[1] + ROI[3]), ROI[0]:(ROI[0] + ROI[2])]
pTau = pTau[crop_ROI[1]:(crop_ROI[1] + crop_ROI[3]), crop_ROI[0]:(crop_ROI[0] + crop_ROI[2])]

chmp2b = sk.io.imread("../data/Dataset 1/round 001/AW GVB AM1c-s11 010426_A01_w4.tif")
chmp2b = chmp2b[ROI[1]:(ROI[1] + ROI[3]), ROI[0]:(ROI[0] + ROI[2])]
chmp2b = chmp2b[crop_ROI[1]:(crop_ROI[1] + crop_ROI[3]), crop_ROI[0]:(crop_ROI[0] + crop_ROI[2])]

# --- Round 2 --- #
pMark = sk.io.imread("../data/Dataset 1/round 002/AM1c-s11-r002_A01_w3.tif")
pMark = pMark[ROI2[1]:(ROI2[1] + ROI2[3]), ROI2[0]:(ROI2[0] + ROI2[2])]

pMark_corr = register.shift_image(pMark, results["shift"])
pMark_corr = pMark_corr[crop_ROI[1]:(crop_ROI[1] + crop_ROI[3]), crop_ROI[0]:(crop_ROI[0] + crop_ROI[2])]

SK1delta = sk.io.imread("../data/Dataset 1/round 002/AM1c-s11-r002_A01_w4.tif")
SK1delta = SK1delta[ROI2[1]:(ROI2[1] + ROI2[3]), ROI2[0]:(ROI2[0] + ROI2[2])]

SK1delta_corr = register.shift_image(SK1delta, results["shift"])
SK1delta_corr = SK1delta_corr[crop_ROI[1]:(crop_ROI[1] + crop_ROI[3]), crop_ROI[0]:(crop_ROI[0] + crop_ROI[2])]

## Find synuclein positive cells
alpha_synuclein_disp = sk.exposure.rescale_intensity(alpha_synuclein, in_range=(0, 2500), out_range=(0.0, 1.0))

mask = alpha_synuclein > 600
mask = sk.morphology.remove_small_objects(mask, max_size=70)
overlay = sk.segmentation.mark_boundaries(alpha_synuclein_disp, mask, mode="thick")

plt.imshow(overlay)
plt.show()
plt.title("Positive cells")
plt.close()

# Find spots in these cells
dog_image = segment.difference_of_gaussians(chmp2b, d_min=3, d_max=20)

mask_spots = dog_image > 0.001
mask_spots_filt = mask_spots & mask


overlay = sk.segmentation.mark_boundaries(sk.exposure.equalize_hist(chmp2b), mask_spots_filt, mode="thick", color=(1.0, 0, 1.0))
plt.imshow(overlay)
plt.title("Spots in chmp2b channel")
plt.show()

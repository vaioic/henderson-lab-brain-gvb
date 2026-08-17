from pathlib import Path

import numpy as np
import skimage
from matplotlib import pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

crop_image = True
ROI = [6310, 628, 9664, 2795]

# Files
lamp1_file = Path(
    r"../test/warped_dataset_output_4555_2/warped_AM1c-s11-r002_A01_channel0.tif"
)

psyn_file = Path(
    r"../processed/20260814_registered_images/AW GVB AM1c-s11 010426_Plate_4536_registered/AW GVB AM1c-s11 010426_A01_channel0.tif"
)


def get_random_color():
    return np.random.randint(0, 256, size=3, dtype=np.uint8)


# DAPI_image = skimage.io.imread(
#     r"../processed/shading_corrected/AM1c-s11-r002_Plate_4555_shifted/AM1c-s11-r002_A01_channel2_channel2.tif"
# )

# Load images
cellbody_image = skimage.io.imread(lamp1_file)

if crop_image:
    cellbody_image = cellbody_image[ROI[1] : ROI[3], ROI[0] : ROI[2]]

alpha_syn_image = skimage.io.imread(psyn_file)

if crop_image:
    alpha_syn_image = alpha_syn_image[ROI[1] : ROI[3], ROI[0] : ROI[2]]

# Try plotting both images to ensure that they are registered correctly
h, w = alpha_syn_image.shape[:2]

rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
cellbody_image_uint8 = skimage.exposure.rescale_intensity(
    cellbody_image, out_range=(0, 255)
).astype(np.uint8)
alpha_syn_image_uint8 = skimage.exposure.rescale_intensity(
    alpha_syn_image, out_range=(0, 255)
).astype(np.uint8)

rgb_image[..., 0] = cellbody_image_uint8
rgb_image[..., 1] = alpha_syn_image_uint8
rgb_image[..., 2] = cellbody_image_uint8

# Set up the Segment Anything model

checkpoint_path = "sam_model/sam_vit_b_01ec64.pth"
model_type = "vit_b"

sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
device = "cpu"
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

# predictor = SamPredictor(sam)

# Cell body only
img_uint8 = np.stack(
    (cellbody_image_uint8, cellbody_image_uint8, cellbody_image_uint8), axis=-1
)
masks = mask_generator.generate(img_uint8)

# plot the masks
output_image = rgb_image.copy()
alpha = 0.3
for idx in range(len(masks)):
    mask = masks[idx]["segmentation"]
    mask_color = get_random_color()

    blended_pixels = (1 - alpha) * output_image[mask] + alpha * mask_color
    output_image[mask] = blended_pixels.astype(np.uint8)

plt.imshow(output_image)
plt.show()

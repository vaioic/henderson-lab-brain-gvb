from pathlib import Path

import numpy as np
import skimage

# Import CellposeModel directly as per the modern API guidelines
from cellpose.models import CellposeModel
from matplotlib import pyplot as plt

crop_image = False
ROI = [6310, 628, 9664, 2795]

# Files
lamp1_file = Path(
    r"../test/warped_dataset_output_4555_2/warped_AM1c-s11-r002_A01_channel0.tif"
)
dapi_file = Path(
    r"../test/warped_dataset_output_4555_2/warped_AM1c-s11-r002_A01_channel1.tif"
)

psyn_file = Path(
    r"../processed/20260814_registered_images/AW GVB AM1c-s11 010426_Plate_4536_registered/AW GVB AM1c-s11 010426_A01_channel0.tif"
)


def get_random_color():
    return np.random.randint(0, 256, size=3, dtype=np.uint8)


# Load images
cellbody_image = skimage.io.imread(lamp1_file)
if crop_image:
    cellbody_image = cellbody_image[ROI[1] : ROI[3], ROI[0] : ROI[2]]

alpha_syn_image = skimage.io.imread(psyn_file)
if crop_image:
    alpha_syn_image = alpha_syn_image[ROI[1] : ROI[3], ROI[0] : ROI[2]]

dapi_image = skimage.io.imread(dapi_file)
if crop_image:
    dapi_image = dapi_image[ROI[1] : ROI[3], ROI[0] : ROI[2]]

# Preprocess & stretch contrast
p_low, p_high = np.percentile(cellbody_image, (2, 98))
cellbody_image_uint8 = skimage.exposure.rescale_intensity(
    cellbody_image, in_range=(p_low, p_high), out_range=(0, 255)
).astype(np.uint8)

p_low, p_high = np.percentile(dapi_image, (2, 98))
dapi_image_uint8 = skimage.exposure.rescale_intensity(
    dapi_image, in_range=(p_low, p_high), out_range=(0, 255)
).astype(np.uint8)

alpha_syn_image_uint8 = skimage.exposure.rescale_intensity(
    alpha_syn_image,
    out_range=(0, 255),
).astype(np.uint8)

# Create background RGB composite for visualization
h, w = alpha_syn_image.shape[:2]
rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
rgb_image[..., 0] = cellbody_image_uint8
rgb_image[..., 1] = cellbody_image_uint8
rgb_image[..., 2] = dapi_image_uint8

# --- Modern Cellpose Initialization ---
# Use CellposeModel and pass the preferred model name directly as a string.
# Available cutting-edge models: 'cpsam_v2' (SAM-ViTL based) or 'cpdino' (DINOv3-ViTL based)
model = CellposeModel(gpu=False)  # , model_type="cpsam_v2")

# --- Inference via CellposeModel.eval ---
# Note: 'channels' parameter maps grayscale single-channel to [0, 0]
masks, flows, styles = model.eval(
    rgb_image,
    # channels=[0, 0],
    flow_threshold=0.8,
    cellprob_threshold=-1.0,
)

# --- Mask Overlay Generation ---
output_image = rgb_image.copy()
alpha = 0.3

# Unique IDs from the output mask array (skipping index 0 which maps background)
unique_mask_ids = np.unique(masks)[1:]

output_dir = Path(r"../processed/2026-08-17 Dev")
output_dir.mkdir(exist_ok=True, parents=True)

skimage.io.imsave(output_dir / "cell_masks.tif", masks)

for cell_id in unique_mask_ids:
    mask = masks == cell_id
    mask_color = get_random_color()

    blended_pixels = (1 - alpha) * output_image[mask] + alpha * mask_color
    output_image[mask] = blended_pixels.astype(np.uint8)

skimage.io.imsave(output_dir / "overlay_cells.png", output_image)
exit()

plt.figure(figsize=(10, 10))
plt.imshow(output_image)
plt.axis("off")
plt.show()

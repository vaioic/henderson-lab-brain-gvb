from pathlib import Path

import numpy as np
import skimage
from matplotlib import pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from skimage.draw import rectangle_perimeter

crop_image = True
ROI = [6310, 628, 9664, 2795]

# Files
lamp1_file = Path(
    r"../test/warped_dataset_output_4555_2/warped_AM1c-s11-r002_A01_channel0.tif"
)

psyn_file = Path(
    r"../processed/20260814_registered_images/AW GVB AM1c-s11 010426_Plate_4536_registered/AW GVB AM1c-s11 010426_A01_channel0.tif"
)


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

# plt.imshow(rgb_image)
# plt.show()

# exit()

# Try to find the pSyn positive cells

rough_cell_mask = cellbody_image > 400  # 0.4 if normalizing
rough_cell_mask = skimage.morphology.opening(
    rough_cell_mask, skimage.morphology.disk(3)
)

rough_cell_mask = skimage.morphology.remove_small_holes(rough_cell_mask, max_size=500)

# Find regions that are positive
alpha_syn_mask = alpha_syn_image > 550

alpha_syn_mask = skimage.morphology.remove_small_objects(alpha_syn_mask, max_size=100)
alpha_syn_mask = alpha_syn_mask & rough_cell_mask

# import oic_toolkit

# ov = oic_toolkit.display.overlay_mask(rgb_image, alpha_syn_mask)
# plt.imshow(ov)
# plt.show()
# exit()

# Get the ROI around the positive markers
alpha_syn_labels = skimage.measure.label(alpha_syn_mask)
props = skimage.measure.regionprops(alpha_syn_labels)

min_area_pixels = 50
valid_boxes = []

box_size = 25

for prop in props:
    if prop.area >= min_area_pixels:
        # skimage output: (ymin, xmin, ymax, xmax)
        ymin, xmin, ymax, xmax = prop.bbox

        # Convert to SAM format: [xmin, ymin, xmax, ymax]
        sam_box = np.array(
            [xmin - box_size, ymin - box_size, xmax + box_size, ymax + box_size]
        )
        valid_boxes.append(sam_box)

# # Draw boxes to check
# output_image = rgb_image.copy()
# box_color = (255, 255, 255)
# for box in valid_boxes:
#     xmin, ymin, xmax, ymax = box

#     rr, cc = rectangle_perimeter(
#         start=(ymin, xmin),
#         end=(ymax - 1, xmax - 1),
#         shape=output_image.shape[:2],
#         clip=True,
#     )
#     output_image[rr, cc] = box_color


# plt.imshow(output_image)
# plt.show()
# exit()

# Set up the Segment Anything model

checkpoint_path = "sam_model/sam_vit_b_01ec64.pth"
model_type = "vit_b"

sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
device = "cpu"
sam.to(device=device)

predictor = SamPredictor(sam)

# Cell body only
img_uint8 = np.stack(
    (cellbody_image_uint8, cellbody_image_uint8, cellbody_image_uint8), axis=-1
)

predictor.set_image(img_uint8)

predicted_masks = []

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(cellbody_image)
ax.axis("off")


def get_random_color():
    return np.random.randint(0, 256, size=3, dtype=np.uint8)


box_color = np.array([255, 255, 255], dtype=np.uint8)
alpha = 0.4  # Transparency factor (40% color blend)

output_image = rgb_image.copy()

for box in valid_boxes:
    # Generate mask for each positive region bounding box
    masks, scores, logits = predictor.predict(
        box=box,
        multimask_output=True,
    )

    # Return the smallest mask
    mask_areas = [np.sum(m) for m in masks]
    smallest_idx = np.argmin(mask_areas)
    smallest_mask = masks[smallest_idx]

    # plt.imshow(smallest_mask)
    # plt.show()
    # exit()

    # masks is a boolean array of shape (1, H, W)
    mask = smallest_mask
    mask_color = get_random_color()

    blended_pixels = (1 - alpha) * output_image[mask] + alpha * mask_color
    output_image[mask] = blended_pixels.astype(np.uint8)

    # output_image[mask] = (1 - alpha) * output_image[mask] + alpha * mask_color

    xmin, ymin, xmax, ymax = box

    rr, cc = rectangle_perimeter(
        start=(ymin, xmin),
        end=(ymax - 1, xmax - 1),
        shape=output_image.shape[:2],
        clip=True,
    )
    output_image[rr, cc] = box_color

skimage.io.imsave("segmentation.png", output_image)

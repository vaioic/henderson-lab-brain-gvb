from pathlib import Path

import oic_toolkit
import skimage

from shared import gvb_analyzer

data_dir = Path("../processed/shading_corrected/")

# Target
dataset1_dir = Path("AW GVB AM1c-s11 010426_Plate_4536_tiled")
dataset1_fn = "AW GVB AM1c-s11 010426_A01_channel"

# Moving
dataset2_dir = Path("AM1c-s11-r002_Plate_4555_tiled")
dataset2_fn = "AM1c-s11-r002_A01_channel"

# Register the images using the DAPI channels
img1_dapi = skimage.io.imread(data_dir / dataset1_dir / (dataset1_fn + "2.tif"))
img2_dapi = skimage.io.imread(data_dir / dataset2_dir / (dataset2_fn + "2.tif"))

results, _, _ = oic_toolkit.register.phasexcorr(img1_dapi, img2_dapi)

# Save the shift to a text file
output_path_moving = Path(
    r"..\processed\registered_images\AM1c-s11-r002_Plate_4555_shifted"
)
output_path_moving.mkdir(exist_ok=True, parents=True)

shift = results["shift"]

shift_str = ", ".join(map(str, shift))
with open(output_path_moving / "registered_shift.txt", "w") as file:
    file.write(shift_str)


# --- Crop the images ---
import numpy as np

# Moving
moving_image_stack = []
for iC in range(4):
    fp = data_dir / dataset2_dir / (dataset2_fn + f"{iC + 1}.tif")
    moving_image_stack.append(skimage.io.imread(fp))

moving_image_stack = np.stack(moving_image_stack, axis=-1)

# Target
target_image_stack = []
for iC in range(4):
    fp = data_dir / dataset1_dir / (dataset1_fn + f"{iC + 1}.tif")
    target_image_stack.append(skimage.io.imread(fp))

target_image_stack = np.stack(target_image_stack, axis=-1)

# Shift and crop the images
target_cropped, moving_cropped = oic_toolkit.register.crop_overlap(
    target_image_stack, moving_image_stack, shift
)

output_path_target = Path(
    r"../processed/20260814_registered_images/AW GVB AM1c-s11 010426_Plate_4536_registered"
)
output_path_target.mkdir(exist_ok=True, parents=True)

for iC in range(target_cropped.shape[-1]):
    skimage.io.imsave(
        output_path_target / f"AW GVB AM1c-s11 010426_A01_channel{iC}.tif",
        target_cropped[..., iC],
    )

output_path_moving = Path(
    r"../processed/20260814_registered_images/AM1c-s11-r002_Plate_4555_registered"
)
output_path_moving.mkdir(exist_ok=True, parents=True)

for iC in range(moving_cropped.shape[-1]):
    skimage.io.imsave(
        output_path_moving / f"AM1c-s11-r002_A01_channel{iC}.tif",
        moving_cropped[..., iC],
    )

gvb_analyzer.merge_tiffs(
    [
        output_path_moving,
        output_path_target,
    ],
    r"../processed/20260814_mergedImages",
)

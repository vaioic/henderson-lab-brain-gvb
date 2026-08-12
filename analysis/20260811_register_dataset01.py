from pathlib import Path

import oic_toolkit
import skimage

data_dir = Path("../processed/shading_corrected/")

dataset1_dir = Path("AW GVB AM1c-s11 010426_Plate_4536")
dataset1_fn = "AW GVB AM1c-s11 010426_A01_channel"

dataset2_dir = Path("AM1c-s11-r002_Plate_4555")
dataset2_fn = "AM1c-s11-r002_A01_channel"

# # Register the images using the DAPI channels
# img1_dapi = skimage.io.imread(data_dir / dataset1_dir / (dataset1_fn + "2.tif"))
# img2_dapi = skimage.io.imread(data_dir / dataset2_dir / (dataset2_fn + "2.tif"))

# results, img2_dapi_corr, img1_dapi_cropped = oic_toolkit.register.phasexcorr(
#     img1_dapi, img2_dapi
# )

# output_path = Path(r"..\processed\shading_corrected\AM1c-s11-r002_Plate_4555_shifted")
# output_path.mkdir(exist_ok=True, parents=True)

# # Save the shift to a text file
# shift = results["shift"]

# shift_str = ", ".join(map(str, shift))
# with open(output_path / "registered_shift.txt", "w") as file:
#     file.write(shift_str)

# ## Process the moving image channels
# for iC in range(4):
#     fp = data_dir / dataset2_dir / (dataset2_fn + f"{iC + 1}.tif")
#     curr_img = skimage.io.imread(fp)
#     corrected = oic_toolkit.register.shift_image(curr_img, results["shift"])
#     skimage.io.imsave(output_path / f"{fp.stem}_channel{iC + 1}.tif", corrected)


results = {"shift": (1556.2, 808.0)}

output_path = Path(
    r"..\processed\shading_corrected\AW GVB AM1c-s11 010426_Plate_4536_shifted2"
)
output_path.mkdir(exist_ok=True, parents=True)

## Process the target image channels
for iC in range(4):
    fp = data_dir / dataset1_dir / (dataset1_fn + f"{iC + 1}.tif")
    curr_img = skimage.io.imread(fp)
    corrected = oic_toolkit.register.shift_image(
        curr_img,
        results["shift"],
        image_type="target",
        tmp_shape=(15968, 12488),
    )
    skimage.io.imsave(output_path / f"{fp.stem}_channel{iC + 1}.tif", corrected)

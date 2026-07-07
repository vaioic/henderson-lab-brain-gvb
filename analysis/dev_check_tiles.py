# The W# indicates a different channel
from pathlib import Path

import numpy as np
import skimage
from natsort import natsorted

# image1 = skimage.io.imread(
#     r"D:\Projects\henderson-lab-brain-gvb\data\Dataset 1\raw\AM1c-s11-r002_Plate_4555\TimePoint_1\\AM1c-s11-r002_A01_s18_w1.TIF"
# )

# image2 = skimage.io.imread(
#     r"D:\Projects\henderson-lab-brain-gvb\data\Dataset 1\raw\AM1c-s11-r002_Plate_4555\TimePoint_1\\AM1c-s11-r002_A01_s18_w2.TIF"
# )

# image3 = skimage.io.imread(
#     r"D:\Projects\henderson-lab-brain-gvb\data\Dataset 1\raw\AM1c-s11-r002_Plate_4555\TimePoint_1\\AM1c-s11-r002_A01_s18_w3.TIF"
# )

# image4 = skimage.io.imread(
#     r"D:\Projects\henderson-lab-brain-gvb\data\Dataset 1\raw\AM1c-s11-r002_Plate_4555\TimePoint_1\\AM1c-s11-r002_A01_s18_w4.TIF"
# )


# fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# axes[0, 0].imshow(image1)
# axes[0, 1].imshow(image2)
# axes[1, 0].imshow(image3)
# axes[1, 1].imshow(image4)
# plt.show()

# Try to get correction

input = Path(
    r"D:\Projects\henderson-lab-brain-gvb\data\Dataset 1\raw\AW GVB AM1c-s11 010426_Plate_4536\TimePoint_1"
)

image_list = list(input.glob("*_w2.TIF"))
image_list = natsorted(image_list)

print(image_list[:10])
exit()

print(f"Found {len(image_list)} files.")

image = skimage.io.imread(image_list[0])

# Create a matrix to hold images
image_data = np.zeros((image.shape[0], image.shape[1], len(image_list)))

for idx, file in enumerate(image_list):
    image_data[:, :, idx] = skimage.io.imread(file)

# Compute the mean
mean_data = np.median(image_data, axis=-1)

# Blur the image
mean_data_blurred = skimage.filters.gaussian(mean_data, sigma=50)


# plt.imshow(mean_data_blurred)
# plt.show()

output = Path("../processed/20260706_test_illumCorr_AW GVB AM1c-s11 010426_Plate_4536")

skimage.io.imsave(""

# Test the correction
for idx in range(image_data.shape[2]):
    test_image = (image_data[:, :, idx]).astype("float") / mean_data_blurred

    skimage.io.imsave(output / f"img_{(idx + 1):02}.tif", test_image)

# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(image_data[:, :, 18])
# axes[1].imshow(test_image)
# plt.show()

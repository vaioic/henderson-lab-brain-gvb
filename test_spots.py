from core import core_funcs
from matplotlib import pyplot as plt

ROI = [8866, 1979, 3000, 3000]
images = core_funcs.import_image_stack("../data/Dataset 1/round 001/", roi=ROI)
# images2 = core_funcs.import_image_stack("../data/Dataset 1/round 002/")

print(images.shape)

plt.imshow(images[1, ...])
plt.show()

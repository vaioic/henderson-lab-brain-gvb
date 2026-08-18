from matplotlib import pyplot as plt

from shared import gvb_analyzer

image_dict = {
    "LAMP1": r"..\processed\20260814_registered_images\AW GVB AM1c-s11 010426_Plate_4536_registered\AW GVB AM1c-s11 010426_A01_channel0.tif",
    "DAPI1": r"..\processed\20260814_registered_images\AW GVB AM1c-s11 010426_Plate_4536_registered\AW GVB AM1c-s11 010426_A01_channel1.tif",
    "pMARK": r"..\processed\20260814_registered_images\AW GVB AM1c-s11 010426_Plate_4536_registered\AW GVB AM1c-s11 010426_A01_channel2.tif",
    "CK1delta": r"..\processed\20260814_registered_images\AW GVB AM1c-s11 010426_Plate_4536_registered\AW GVB AM1c-s11 010426_A01_channel3.tif",
    "pSyn": r"../test/warped_dataset_output_4555_2/warped_AM1c-s11-r002_A01_channel0.tif",
    "DAPI2": r"../test/warped_dataset_output_4555_2/warped_AM1c-s11-r002_A01_channel1.tif",
    "pTau": r"../test/warped_dataset_output_4555_2/warped_AM1c-s11-r002_A01_channel2.tif",
    "CHMP2B": r"../test/warped_dataset_output_4555_2/warped_AM1c-s11-r002_A01_channel3.tif",
}

manager = gvb_analyzer.ImageManager(image_dict)

img = manager.get_image("DAPI1")

plt.imshow(img)
plt.show()

import numpy as np
import skimage as sk
from matplotlib import pyplot as plt
from scipy.ndimage import shift as ndimage_shift


def register_images(target, moving):

    h0, w0 = target.shape
    hm, wm = moving.shape

    # Match the moving to the target shape
    moving_final = np.zeros_like(target)

    # Determine the overlapping bounds
    slice_y = min(h0, hm)
    slice_x = min(w0, wm)

    moving_final[:slice_y, :slice_x] = moving[:slice_y, :slice_x]

    plot_merged_images(target, moving_final)

    # Run the cross-correlation to register the images
    shift, error, diffphase = sk.registration.phase_cross_correlation(
        target, 
        moving_final, 
        disambiguate=True
    )
    
    moving_corrected = ndimage_shift(
        moving_final, 
        shift=shift, 
        cval=0.0
    )

    plot_merged_images(target, moving_corrected)
    

def plot_merged_images(target, moving):

    merged = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.uint8)

    merged[..., 0] = target
    merged[..., 1] = moving
    merged[..., 2] = target

    plt.imshow(merged)
    plt.show()


I1 = sk.io.imread("../data/Dataset 1/round 001/AW GVB AM1c-s11 010426_A01_w2.tif")

I2 = sk.io.imread("../data/Dataset 1/round 002/AM1c-s11-r002_A01_w2.tif")

print(I1.shape)
print(I1.dtype)
print(I2.shape)

register_images(I1, I2)




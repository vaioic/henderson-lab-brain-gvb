import glob
import os

import SimpleITK as sitk

# =========================================================================
# STEP 1: LOAD THE PRIMARY IMAGES (To calculate the grid deformation)
# =========================================================================
# 'fixed' stays still. 'moving' is the image that needs to be warped.
fixed_image = sitk.ReadImage(
    r"D:\Projects\henderson-lab-brain-gvb\processed\20260814_registered_images\AW GVB AM1c-s11 010426_Plate_4536_registered\AW GVB AM1c-s11 010426_A01_channel1.tif",
    sitk.sitkFloat32,
)
moving_image = sitk.ReadImage(
    r"D:\Projects\henderson-lab-brain-gvb\processed\20260814_registered_images\AM1c-s11-r002_Plate_4555_registered\AM1c-s11-r002_A01_channel1.tif",
    sitk.sitkFloat32,
)


# =========================================================================
# STEP 2: CONFIGURE THE GRID (Dividing the image into local sub-patches)
# =========================================================================
# 'transformDomainMeshSize' defines the grid layout.
# For example, [8, 8] splits the image into an 8x8 grid of local patches.
# Increase these numbers (e.g., [12, 12]) for smaller subimages / more localized warping.
grid_size = [32, 32]
bspline_transform = sitk.BSplineTransformInitializer(
    fixed_image, transformDomainMeshSize=grid_size
)


# =========================================================================
# STEP 3: CONFIGURE AND RUN THE REGISTRATION ENGINE
# =========================================================================
registration = sitk.ImageRegistrationMethod()
registration.SetInitialTransform(bspline_transform, inPlace=False)

# Metric: Evaluates how well the pixel intensities match between subimages
registration.SetMetricAsMeanSquares()
registration.SetMetricSamplingStrategy(registration.RANDOM)
registration.SetMetricSamplingPercentage(
    0.15
)  # Samples 15% of pixels to speed up calculation

# Optimizer: Controls how the grid corners are tweaked to minimize alignment error
registration.SetOptimizerAsLBFGSB(
    gradientConvergenceTolerance=1e-5, numberOfIterations=100
)
registration.SetInterpolator(sitk.sitkLinear)

print("Dividing image into grids and calculating local transformations...")
calculated_transform = registration.Execute(fixed_image, moving_image)


# =========================================================================
# STEP 4: SAVE THE TRANSFORMATION MATRIX FILE
# =========================================================================
# This saves the exact grid math/deformation map to your hard drive.
transform_file_path = "dataset_grid_warp.tfm"
sitk.WriteTransform(calculated_transform, transform_file_path)
print(f"Transformation matrix successfully saved to: {transform_file_path}")


# =========================================================================
# STEP 5: BATCH-APPLY THE SAME TRANSFORMATION TO YOUR DATASET
# =========================================================================
# Load the saved transformation map back from disk
saved_transform = sitk.ReadTransform(transform_file_path)

# Specify the folder containing the other images you want to warp
input_folder = r"D:\Projects\henderson-lab-brain-gvb\processed\20260814_registered_images\AM1c-s11-r002_Plate_4555_registered"
output_folder = "../test/warped_dataset_output_4555_2/"
os.makedirs(output_folder, exist_ok=True)

# Find all matching images in your dataset (adjust extension like .tif, .png as needed)
dataset_images = glob.glob(os.path.join(input_folder, "*.tif"))

print(f"Found {len(dataset_images)} images to process. Starting batch warping...")

for img_path in dataset_images:
    filename = os.path.basename(img_path)

    # Read the dataset image
    current_img = sitk.ReadImage(img_path, sitk.sitkFloat32)

    # Apply the exact same grid transform instantly without recalculating anything
    warped_img = sitk.Resample(
        current_img,
        fixed_image,  # Defines the output grid size, spacing, and origin
        saved_transform,  # The grid transformation map we calculated earlier
        sitk.sitkLinear,  # Interpolator type
        0.0,  # Default pixel value for areas pushed outside the frame
        current_img.GetPixelID(),
    )

    # Save the newly aligned image
    output_path = os.path.join(output_folder, f"warped_{filename}")
    sitk.WriteImage(warped_img, output_path)
    print(f" -> Processed and saved: {output_path}")

print("All images in the dataset have been successfully warped!")

import os

import cv2
import numpy as np


def get_exact_shift(tile_a, tile_b, estimated_overlap):
    """
    Uses OpenCV template matching to find the exact pixel offset,
    with built-in guardrails against negative or impossible values.
    Casts to float32 to fully support 16-bit scientific TIFF profiles.
    """
    h, w = tile_a.shape[:2]

    # Define a reasonable search window
    template_w = int(estimated_overlap * 0.8)
    template = tile_a[:, w - template_w :]
    search_region = tile_b[:, : int(estimated_overlap * 1.5)]

    # Cast to float32 for 16-bit imaging compatibility
    template_32f = template.astype(np.float32)
    search_region_32f = search_region.astype(np.float32)

    # Calculate cross-correlation match template matrix
    res = cv2.matchTemplate(search_region_32f, template_32f, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)

    # Match location relative to the original overlap boundary
    exact_overlap = w - (max_loc[0] - (w - template_w))

    # Safe limits: Force overlap evaluation between 50% and 150% of the estimate
    min_acceptable = int(estimated_overlap * 0.5)
    max_acceptable = int(estimated_overlap * 1.5)

    if not (min_acceptable <= exact_overlap <= max_acceptable):
        return estimated_overlap

    return exact_overlap


def opencv_overlap_stitch(image_folder, x_sites, y_sites, overlap_ratio=0.15):
    """
    Stitches a grid of zero-padded sequential image tiles using an
    alpha-feathering ramp to eliminate seam artifacts and exposure stripes.
    """
    valid_exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    file_names = sorted(
        [f for f in os.listdir(image_folder) if f.lower().endswith(valid_exts)]
    )

    if len(file_names) < (x_sites * y_sites):
        raise ValueError(
            f"Found {len(file_names)} images. Needed at least {x_sites * y_sites}."
        )

    # 1. Read first tile to map dimensions, structural channels, and bit depth
    first_tile_path = os.path.join(image_folder, file_names[0])
    first_tile = cv2.imread(first_tile_path, cv2.IMREAD_UNCHANGED)
    tile_h, tile_w = first_tile.shape[:2]
    dtype = first_tile.dtype
    is_color = len(first_tile.shape) == 3

    est_overlap_x = int(tile_w * overlap_ratio)
    est_overlap_y = int(tile_h * overlap_ratio)

    # 2. Build 2D list array of image matrices
    grid = []
    idx = 0
    for y in range(y_sites):
        row = []
        for x in range(x_sites):
            img_path = os.path.join(image_folder, file_names[idx])
            row.append(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))
            idx += 1
        grid.append(row)

    # 3. Calculate optimized overlapping pixel margins using adjacent tiles
    real_overlap_x = get_exact_shift(grid[0][0], grid[0][1], est_overlap_x)
    real_overlap_y = get_exact_shift(grid[0][0], grid[1][0], est_overlap_y)

    step_w = tile_w - real_overlap_x
    step_h = tile_h - real_overlap_y

    # 4. Compute maximum theoretical grid bounds and allocate empty canvas frames
    canvas_h = step_h * (y_sites - 1) + tile_h
    canvas_w = step_w * (x_sites - 1) + tile_w

    if not is_color:
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        weight_canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    else:
        channels = first_tile.shape[2]
        canvas = np.zeros((canvas_h, canvas_w, channels), dtype=np.float32)
        weight_canvas = np.zeros((canvas_h, canvas_w, channels), dtype=np.float32)

    # 5. Pre-build Linear Alpha Ramp Blending Masks
    ramp_y = np.ones(tile_h, dtype=np.float32)
    ramp_x = np.ones(tile_w, dtype=np.float32)

    if real_overlap_y > 0:
        ramp_y[:real_overlap_y] = np.linspace(0, 1, real_overlap_y)
        ramp_y[-real_overlap_y:] = np.linspace(1, 0, real_overlap_y)
    if real_overlap_x > 0:
        ramp_x[:real_overlap_x] = np.linspace(0, 1, real_overlap_x)
        ramp_x[-real_overlap_x:] = np.linspace(1, 0, real_overlap_x)

    tile_mask = np.outer(ramp_y, ramp_x)
    if is_color:
        tile_mask = np.expand_dims(tile_mask, axis=2)

    # 6. Multi-layer map iteration and normalization
    for y in range(y_sites):
        for x in range(x_sites):
            tile = grid[y][x].astype(np.float32)
            y_start = y * step_h
            x_start = x * step_w

            # Blend current tile matrix payload & profile footprint into target areas
            canvas[y_start : y_start + tile_h, x_start : x_start + tile_w] += (
                tile * tile_mask
            )
            weight_canvas[y_start : y_start + tile_h, x_start : x_start + tile_w] += (
                tile_mask
            )

    # Clean zero divisions from unused boundaries
    weight_canvas[weight_canvas == 0] = 1.0
    final_canvas = canvas / weight_canvas

    # Replace the final return line with this:
    if np.issubdtype(dtype, np.integer):
        max_val = np.iinfo(dtype).max
    else:
        max_val = 1.0

    return np.clip(final_canvas, 0, max_val).astype(dtype)


# --- Runtime Execution Example ---
# out_img = opencv_overlap_stitch("./my_tiles_folder", x_sites=7, y_sites=9, overlap_ratio=0.10)
# cv2.imwrite("final_feathered_mosaic.tif", out_img)

# Usage:
stitched = opencv_overlap_stitch(
    r"D:\Projects\henderson-lab-brain-gvb\processed\20260706_test_illumCorr_AW GVB AM1c-s11 010426_Plate_4536",
    x_sites=7,
    y_sites=10,
    overlap_ratio=0.10,
)
cv2.imwrite("stitched_dynamic.tif", stitched)

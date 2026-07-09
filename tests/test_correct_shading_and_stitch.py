# ga.correct_shading(
#     r"D:\Projects\henderson-lab-brain-gvb\data\Dataset 1\raw\AM1c-s11-r002_Plate_4555\TimePoint_1",
#     r"D:\Projects\henderson-lab-brain-gvb\processed\20260707_AM1c-s11-r002_Plate_4555",
#     file_pattern="*w2.TIF",
# )
from pathlib import Path

import pandas as pd
from oic_toolkit.io import export_pyramid_tiff
from oic_toolkit.register import generate_tiled_image, stitch_xy

from shared import gvb_analyzer as ga

output_dir = Path(
    r"D:\Projects\henderson-lab-brain-gvb\processed\shading_corrected\AW GVB AM1c-s11 010426_Plate_4536"
)

ga.correct_all(
    r"D:\Projects\henderson-lab-brain-gvb\data\Dataset 1\raw\AW GVB AM1c-s11 010426_Plate_4536\TimePoint_1",
    output_dir,
)
exit()

# output_dir = Path(
#     r"D:\Projects\henderson-lab-brain-gvb\processed\shading_corrected\AM1c-s11-r002_Plate_4555"
# )

# ga.correct_all(
#     r"D:\Projects\henderson-lab-brain-gvb\data\Dataset 1\raw\AM1c-s11-r002_Plate_4555\TimePoint_1",
#     output_dir,
# )


# abs_x, abs_y = stitch_xy(
#     image_path=output_dir / "AM1c-s11-r002_A01_channel1", numX=7, numY=9, overlap=15
# )

# stitched_image = generate_tiled_image(
#     output_dir / "AM1c-s11-r002_A01_channel1", abs_x=abs_x, abs_y=abs_y
# )

# # print(stitched_image.dtype)

# plt.imshow(stitched_image)
# plt.show()

# export_pyramid_tiff(output_dir / "AM1c-s11-r002_A01_channel1.ome.tiff", stitched_image)

# ---- Channel 2 ----

abs_x, abs_y = stitch_xy(
    image_path=output_dir / "AM1c-s11-r002_A01_channel2", numX=7, numY=9, overlap=12
)

stitched_image = generate_tiled_image(
    output_dir / "AM1c-s11-r002_A01_channel2", abs_x=abs_x, abs_y=abs_y
)

# print(stitched_image.dtype)

# plt.imshow(stitched_image)
# plt.show()

export_pyramid_tiff(output_dir / "AM1c-s11-r002_A01_channel2.ome.tiff", stitched_image)

# ---- Channel 1 ----

stitched_image = generate_tiled_image(
    output_dir / "AM1c-s11-r002_A01_channel1", abs_x=abs_x, abs_y=abs_y
)

export_pyramid_tiff(output_dir / "AM1c-s11-r002_A01_channel1.ome.tiff", stitched_image)


# ---- Channel 3 ----

stitched_image = generate_tiled_image(
    output_dir / "AM1c-s11-r002_A01_channel3", abs_x=abs_x, abs_y=abs_y
)

# print(stitched_image.dtype)

# plt.imshow(stitched_image)
# plt.show()

export_pyramid_tiff(output_dir / "AM1c-s11-r002_A01_channel3.ome.tiff", stitched_image)

# ---- Channel 4 ----

stitched_image = generate_tiled_image(
    output_dir / "AM1c-s11-r002_A01_channel4", abs_x=abs_x, abs_y=abs_y
)

# print(stitched_image.dtype)

# plt.imshow(stitched_image)
# plt.show()

export_pyramid_tiff(output_dir / "AM1c-s11-r002_A01_channel4.ome.tiff", stitched_image)


tile_indices = list(range(len(abs_x)))

# Build the DataFrame and export to CSV
df = pd.DataFrame({"tile_index": tile_indices, "abs_x": abs_x, "abs_y": abs_y})
df.to_csv(output_dir / "tile_positions.csv", index=False)

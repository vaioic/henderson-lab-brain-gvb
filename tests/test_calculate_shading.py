from shared import gvb_analyzer as ga

# shading = ga.calculate_shading(
#     r"D:\Projects\henderson-lab-brain-gvb\data\Dataset 1\raw\AM1c-s11-r002_Plate_4555\TimePoint_1",
# )

# plt.imshow(shading)
# plt.show()

# ga.correct_shading(
#     r"D:\Projects\henderson-lab-brain-gvb\data\Dataset 1\raw\AM1c-s11-r002_Plate_4555\TimePoint_1",
#     r"D:\Projects\henderson-lab-brain-gvb\processed\20260707_AM1c-s11-r002_Plate_4555",
#     file_pattern="*w2.TIF",
# )

ga.correct_shading(
    r"D:\Projects\henderson-lab-brain-gvb\data\Dataset 1\raw\AM1c-s11-r003_Plate_4642\TimePoint_1",
    r"D:\Projects\henderson-lab-brain-gvb\processed\AM1c-s11-r003_Plate_4642",
    file_pattern="*w2.TIF",
)

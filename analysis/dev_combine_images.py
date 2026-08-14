from shared import gvb_analyzer

gvb_analyzer.merge_tiffs(
    [
        r"../processed/shading_corrected/AW GVB AM1c-s11 010426_Plate_4536_shifted2",
        r"../processed/shading_corrected/AM1c-s11-r002_Plate_4555_shifted",
    ],
    r"../processed/20260814_mergedImages",
)

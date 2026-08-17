from shared import gvb_analyzer

gvb_analyzer.merge_tiffs(
    [
        r"../test/warped_dataset_output_4555_2",
        r"../processed/20260814_registered_images/AW GVB AM1c-s11 010426_Plate_4536_registered",
    ],
    r"../processed/20260815_mergedImages",
    channel_names=[
        "LAMP1 (4555)",
        "DAPI (4555)",
        "pMARK (4555)",
        "CK1delta (4555)",
        "pSyn (4536)",
        "DAPI (4536)",
        "pTau (4536)",
        "CHMP2B (4536)",
    ],
)

def get_samples(DATA_DIR):
    """
    input: 'data/' or 'data', the path of czi files
    e.g. 
    E2f4_CFUe_KO_1-Stitching-01.czi       E2f4_CFUe_WT2_1-Stitching-12.czi
    E2f4_CFUe_KO_2-Stitching-02.czi       E2f4_CFUe_WT2_1_Top-Stitching-13.czi
    E2f4_CFUe_KO_3-Stitching-03.czi       E2f4_CFUe_WT2_2-Stitching-14.czi
    E2f4_CFUe_NoEpo_1-Stitching-04.czi    E2f4_CFUe_WT2_3-Stitching-15.czi
    E2f4_CFUe_NoEpo_2-Stitching-05.czi    E2f4_CFUe_WT3_1-Stitching-16.czi
    E2f4_CFUe_NoEpo_3-Stitching-06.czi    E2f4_CFUe_WT3_1_Top-Stitching-17.czi
    E2f4_CFUe_WT1_1-Stitching-07.czi      E2f4_CFUe_WT3_2-Stitching-18.czi
    E2f4_CFUe_WT1_2-Stitching-08.czi      E2f4_CFUe_WT3_2_Top-Stitching-19.czi
    E2f4_CFUe_WT1_2_Top-Stitching-09.czi  E2f4_CFUe_WT3_3-Stitching-20.czi
    E2f4_CFUe_WT1_3-Stitching-10.czi      E2f4_CFUe_WT3_3_Top-Stitching-21.czi
    E2f4_CFUe_WT1_3_Top-Stitching-11.czi  yung.txt

    output: 
    ['E2f4_CFUe_KO_1-Stitching-01',
     'E2f4_CFUe_KO_2-Stitching-02',
     'E2f4_CFUe_KO_3-Stitching-03',
     'E2f4_CFUe_NoEpo_1-Stitching-04',
     'E2f4_CFUe_NoEpo_2-Stitching-05',
     'E2f4_CFUe_NoEpo_3-Stitching-06',
     'E2f4_CFUe_WT1_1-Stitching-07',
     'E2f4_CFUe_WT1_2-Stitching-08',
     'E2f4_CFUe_WT1_2_Top-Stitching-09',
     'E2f4_CFUe_WT1_3-Stitching-10',
     'E2f4_CFUe_WT1_3_Top-Stitching-11',
     'E2f4_CFUe_WT2_1-Stitching-12',
     'E2f4_CFUe_WT2_1_Top-Stitching-13',
     'E2f4_CFUe_WT2_2-Stitching-14',
     'E2f4_CFUe_WT2_3-Stitching-15',
     'E2f4_CFUe_WT3_1-Stitching-16',
     'E2f4_CFUe_WT3_1_Top-Stitching-17',
     'E2f4_CFUe_WT3_2-Stitching-18',
     'E2f4_CFUe_WT3_2_Top-Stitching-19',
     'E2f4_CFUe_WT3_3-Stitching-20',
     'E2f4_CFUe_WT3_3_Top-Stitching-21']
    """
    import os
    import re
    SAMPLES=os.listdir(DATA_DIR)
    SAMPLES=list(filter(lambda x: x.endswith("czi"), SAMPLES))
    SAMPLES=[re.sub(".czi", "", x) for x in SAMPLES]
    return SAMPLES
from spacenetutilities.labeltools import coreLabelTools as clT


import glob
for img in glob.glob("/mnt/hdd/Datasets/AOI_3_Paris_Train/RGB-PanSharpen/*.tif"):
    new_img_path = img.replace(".tif", ".jpeg")
    print(new_img_path)
    clT.convertGTiffTo8Bit(
        img,
        new_img_path,
        "JPEG"
    )




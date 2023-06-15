import rioxarray
import json
import os

"""# GeoJson to Tif
For processing .tif files with .geoJson files together to train the dataset, first phase of the process is creating 'mask' images. 
Mask images are basically the images that are black and white images which contains with pixel values of [0,1] that is the information of buildings. 
To study on masked images, first phase is to create them. For creating raster based masked images, Rasterio has been used. 
*(Due to the limitations of Google Colab, I am able to use only 300 images)*
"""

def geoJson_to_tif(filename):
    filenamePre = filename[0:28]
    fileNum = filename[28:len(filename) - 8]
    filenameEnd = filename[len(filename) - 8:len(filename)]

    # In this training below, 'Shanghai' dataset from SpaceNet v2 has been used.
    # To call Shanghai raster images and polygons, a directory has been created
    # in Google Drive for Google Colab.

    # rasterfilenamepre = "RGB-PanSharpen_AOI_4_Shanghai_img"
    rasterfilenamepre = "RGB-PanSharpen_AOI_3_Paris_img"

    # Location of raster images in .tiff format
    dirtif = "/mnt/hdd/Datasets/AOI_3_Paris_Train/RGB-PanSharpen/"

    # Location of polygon files in .geojson format
    dirgjson = "/mnt/hdd/Datasets/AOI_3_Paris_Train/geojson/buildings/"

    # Location of the place which created masks will held place.
    dirmasktif = "/mnt/hdd/Datasets/AOI_3_Paris_Train/masktif/"

    try:
        # load in the geojson file
        with open(dirgjson + filename) as igj:
            data = json.load(igj)
        # if GDAL 3+
        crs = data["crs"]["properties"]["name"]
        # crs = "EPSG:4326" # if GDAL 2
        geoms = [feat["geometry"] for feat in data["features"]]

        # Create empty mask raster based on the input raster
        rds = rioxarray.open_rasterio(dirtif + rasterfilenamepre + fileNum + ".tif").isel(band=0)
        rds.values[:] = 1
        rds.rio.write_nodata(0, inplace=True)
        rds.rio.to_raster(dirmasktif + rasterfilenamepre + fileNum + "mask.tif", dtype="uint8")

        # clip the raster to the mask
        clipped = rds.rio.clip(geoms, crs, drop=False)
        clipped.rio.to_raster(dirmasktif + rasterfilenamepre + fileNum + "mask.tif", dtype="uint8")
    except:
        rds = rioxarray.open_rasterio(dirtif + rasterfilenamepre + fileNum + ".tif").isel(band=0)
        rds.values[:] = 0
        rds.rio.write_nodata(0, inplace=True)
        rds.rio.to_raster(dirmasktif + rasterfilenamepre + fileNum + "mask.tif", dtype="uint8")


for filename in os.listdir(
        "/mnt/hdd/Datasets/AOI_3_Paris_Train/geojson/buildings/"):
    if filename.endswith(".geojson"):
        print(os.path.join(
            "/mnt/hdd/Datasets/AOI_3_Paris_Train/geojson/buildings/",
            filename))
        # Create masks using the def above to the directory above.
        geoJson_to_tif(filename)
        continue
    else:
        continue
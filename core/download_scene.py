import os
import geemap

from core.scene_fetcher import (
    initialize_gee,
    fetch_scene
)

# =====================================
# DOWNLOAD SAR SCENE
# =====================================

def download_scene(
    min_lat,
    max_lat,
    min_lon,
    max_lon,
    date
):
    """
    Download Sentinel-1 VV image
    covering the AOI.
    """

    scene_info = fetch_scene(

        min_lat=min_lat,
        max_lat=max_lat,

        min_lon=min_lon,
        max_lon=max_lon,

        date=date
    )

    if scene_info is None:

        return None

    scene = scene_info["scene"]

    aoi = scene_info["aoi"]

    sar = scene.select(
        ["VV", "angle"]
    )



    os.makedirs(
        "outputs",
        exist_ok=True
    )

    output_file = (
        "outputs/sar_scene.tif"
    )

    print(
        "\nDownloading SAR scene..."
    )

    print(
    "\nBands in scene:"
    )

    print(
        sar.bandNames().getInfo()
    )

    if os.path.exists(output_file):
        os.remove(output_file)

    geemap.ee_export_image(

        sar,

        filename=output_file,

        scale=10,

        region=aoi,

        file_per_band=False
    )

    if not os.path.exists(output_file):

        print(
            "\nDownload failed"
        )

        return None

    print(
        "\nSaved:",
        output_file
    )

    return output_file


# =====================================
# TEST
# =====================================

# =====================================
# TEST
# =====================================

if __name__ == "__main__":

    initialize_gee()

    output_file = download_scene(

        min_lat=19.0,
        max_lat=19.1,

        min_lon=72.8,
        max_lon=72.9,

        date="2024-01-15"
    )

    if output_file:

        import rasterio

        src = rasterio.open(
            output_file
        )

        print(
            "\nBands:",
            src.count
        )

        print(
            "Shape:",
            src.read(1).shape
        )

        print(
            "Resolution:",
            src.res
        )

        src.close()

    else:

        print(
            "\nDownload failed"
        )
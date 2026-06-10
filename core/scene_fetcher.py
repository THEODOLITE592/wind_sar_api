import ee
import streamlit as st
import json
import tempfile

def initialize_gee():

    try:
        ee.Number(1).getInfo()
        return
    except:
        pass

    service_account = st.secrets["gcp_service_account"]

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False
    ) as fp:

        json.dump(
            dict(service_account),
            fp
        )

        key_path = fp.name

    credentials = ee.ServiceAccountCredentials(
        service_account["client_email"],
        key_path
    )

    ee.Initialize(
        credentials,
        project="crop-472500"
    )

    print("GEE initialized")

# =====================================
# FETCH SINGLE SENTINEL SCENE
# =====================================

def fetch_scene(

    min_lat,
    max_lat,

    min_lon,
    max_lon,

    date
):
    """
    Fetch ONE Sentinel-1 scene
    covering the entire AOI.

    Parameters
    ----------
    min_lat : float
    max_lat : float

    min_lon : float
    max_lon : float

    date : str
        YYYY-MM-DD

    Returns
    -------
    dict or None

    {
        "scene": ee.Image,
        "aoi": ee.Geometry,
        "scene_id": str,
        "acquisition_date": str
    }
    """

    try:

        # ==========================
        # AOI
        # ==========================

        aoi = ee.Geometry.Rectangle([

            min_lon,
            min_lat,

            max_lon,
            max_lat
        ])

        # ==========================
        # Sentinel Collection
        # ==========================

        collection = (

            ee.ImageCollection(
                "COPERNICUS/S1_GRD"
            )

            .filterBounds(
                aoi
            )

            .filterDate(

                ee.Date(date).advance(
                    -15,
                    "day"
                ),

                ee.Date(date).advance(
                    15,
                    "day"
                )
            )

            .filter(

                ee.Filter.eq(
                    "instrumentMode",
                    "IW"
                )
            )

            .filter(

                ee.Filter.listContains(
                    "transmitterReceiverPolarisation",
                    "VV"
                )
            )

            .select(
                ["VV", "angle"]
            )

            .sort(
                "system:time_start" ,
                 False
            )
        )

        count = (

            collection
            .size()
            .getInfo()
        )

        if count == 0:

            print(
                "No Sentinel scene found"
            )

            return None

        # ==========================
        # Select Scene
        # ==========================

        scene = ee.Image(
            collection.first()
        )

        info = scene.getInfo()

        scene_id = info["id"]

        timestamp = (

            info["properties"]
            ["system:time_start"]
        )

        acquisition_date = (
            ee.Date(timestamp)
            .format(
                "YYYY-MM-dd"
            )
            .getInfo()
        )

        print(
            "\nScene Found"
        )

        print(
            "Scene ID:",
            scene_id
        )

        print(
            "Date:",
            acquisition_date
        )

        return {

            "scene":
                scene,

            "aoi":
                aoi,

            "scene_id":
                scene_id,

            "acquisition_date":
                acquisition_date
        }

    except Exception as e:

        print(
            "Scene fetch failed:",
            e
        )

        return None


# =====================================
# TEST
# =====================================

if __name__ == "__main__":

    initialize_gee()

    result = fetch_scene(

        min_lat=19.0,
        max_lat=19.2,

        min_lon=72.8,
        max_lon=73.0,

        date="2024-01-15"
    )

    if result:

        print(
            "\nSUCCESS"
        )

        print(
            result["scene_id"]
        )

    else:

        print(
            "\nNo scene found"
        )
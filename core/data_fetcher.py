import ee

from core.config import (
    PROJECT_ID,
    PATCH_SIZE_M
)

# =====================================
# INITIALIZE GEE
# =====================================

def initialize_gee():
    """
    Initialize Google Earth Engine.
    Call once at application startup.
    """

    try:

        ee.Initialize(
            project=PROJECT_ID
        )

        print(
            "GEE initialized"
        )

    except Exception:

        ee.Authenticate()

        ee.Initialize(
            project=PROJECT_ID
        )

        print(
            "GEE authenticated and initialized"
        )


# =====================================
# FETCH DATA
# =====================================

def fetch_data(
    lat,
    lon,
    date,
    patch_size_m=PATCH_SIZE_M
):
    """
    Fetch Sentinel-1 SAR patch and ERA5 winds.

    Parameters
    ----------
    lat : float
    lon : float
    date : str
        Format: YYYY-MM-DD

    patch_size_m : int
        Ground patch size in meters.

    Returns
    -------
    dict or None

    {
        "sar_image": ee.Image,
        "patch_geometry": ee.Geometry,
        "u_era5": float,
        "v_era5": float,
        "lat": float,
        "lon": float,
        "date": str
    }
    """

    try:

        # ==============================
        # POINT & PATCH
        # ==============================

        point = ee.Geometry.Point(
            [lon, lat]
        )

        patch = (
            point
            .buffer(
                patch_size_m / 2
            )
            .bounds()
        )

        # ==============================
        # SENTINEL-1
        # ==============================

        s1 = (
            ee.ImageCollection(
                "COPERNICUS/S1_GRD"
            )
            .filterBounds(
                point
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
        )

        count = (
            s1.size()
            .getInfo()
        )

        if count == 0:

            print(
                f"No Sentinel-1 scene found "
                f"for {date}"
            )

            return None

        sar_image = (
            ee.Image(
                s1.first()
            )
            .clip(
                patch
            )
        )

        # ==============================
        # ERA5
        # ==============================

        era5 = (
            ee.ImageCollection(
                "ECMWF/ERA5_LAND/HOURLY"
            )
            .filterDate(
                date,
                ee.Date(date).advance(
                    1,
                    "day"
                )
            )
            .first()
        )

        if era5 is None:

            print(
                f"No ERA5 data found "
                f"for {date}"
            )

            return None

        vals = (
            era5
            .select([
                "u_component_of_wind_10m",
                "v_component_of_wind_10m"
            ])
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=1000,
                maxPixels=1e8
            )
            .getInfo()
        )

        if vals is None:

            return None

        u_era5 = vals.get(
            "u_component_of_wind_10m"
        )

        v_era5 = vals.get(
            "v_component_of_wind_10m"
        )

        if (
            u_era5 is None
            or
            v_era5 is None
        ):
            return None

        # ==============================
        # RETURN
        # ==============================

        return {

            "sar_image":
                sar_image,

            "patch_geometry":
                patch,

            "u_era5":
                float(u_era5),

            "v_era5":
                float(v_era5),

            "lat":
                float(lat),

            "lon":
                float(lon),

            "date":
                date
        }

    except Exception as e:

        print(
            "Fetch failed:",
            e
        )

        return None


# =====================================
# TEST
# =====================================

if __name__ == "__main__":

    initialize_gee()

    result = fetch_data(

        lat=19.0760,
        lon=72.8777,
        date="2024-01-15"
    )

    if result:

        print("\nSUCCESS")

        print(
            "ERA5 U:",
            result["u_era5"]
        )

        print(
            "ERA5 V:",
            result["v_era5"]
        )

    else:

        print(
            "\nNo data found"
        )
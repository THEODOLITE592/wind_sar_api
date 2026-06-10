import ee
import pandas as pd
import numpy as np

from core.scene_fetcher import (
    initialize_gee
)


def get_era5_grid(

    points,
    date
):

    # =====================
    # ERA5 IMAGE
    # =====================

    era5 = (

        ee.ImageCollection(
            "ECMWF/ERA5/HOURLY"
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

    image = era5.select([

        "u_component_of_wind_10m",
        "v_component_of_wind_10m"

    ])

    # =====================
    # FEATURE COLLECTION
    # =====================

    features = []

    for lat, lon in points:

        features.append(

            ee.Feature(

                ee.Geometry.Point(
                    [lon, lat]
                ),

                {
                    "lat": lat,
                    "lon": lon
                }
            )
        )

    fc = ee.FeatureCollection(
        features
    )

    # =====================
    # SAMPLE ONCE
    # =====================

    sampled = image.sampleRegions(

        collection=fc,

        scale=1000,

        geometries=True
    )

    records = sampled.getInfo()

    rows = []

    for f in records["features"]:

        props = f["properties"]

        rows.append({

            "lat":
                props["lat"],

            "lon":
                props["lon"],

            "u":
                props[
                    "u_component_of_wind_10m"
                ],

            "v":
                props[
                    "v_component_of_wind_10m"
                ]
        })

    return pd.DataFrame(
        rows
    )

if __name__ == "__main__":

    initialize_gee()

    points = []

    for lat in np.arange(
        19.0,
        19.1,
        0.01
    ):
        for lon in np.arange(
            72.8,
            72.9,
            0.01
        ):

            points.append(
                (lat, lon)
            )


    df = get_era5_grid(

        points,
        "2024-01-15"
    )

    print(
        "Requested:",
        len(points)
    )

    print(
        "Returned:",
        len(df)
    )

    print(df)
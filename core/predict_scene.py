import os
import pandas as pd
import numpy as np

from core.predict_patch import (
    predict_patch
)

from core.data_fetcher import (
    initialize_gee
)

from core.config import (
    GRID_SPACING_KM
)

# =====================================
# GRID GENERATION
# =====================================

def generate_grid(
    min_lat,
    max_lat,
    min_lon,
    max_lon
):
    """
    Generate grid points.

    Approx:
    1 degree latitude ≈ 111 km
    """

    step_deg = (
        GRID_SPACING_KM / 111.0
    )

    lats = np.arange(
        min_lat,
        max_lat + step_deg,
        step_deg
    )

    lons = np.arange(
        min_lon,
        max_lon + step_deg,
        step_deg
    )

    points = []

    for lat in lats:

        for lon in lons:

            points.append(
                (float(lat), float(lon))
            )

    return points


# =====================================
# PREDICT SCENE
# =====================================

def predict_scene(
    min_lat,
    max_lat,
    min_lon,
    max_lon,
    date
):

    initialize_gee()

    points = generate_grid(

        min_lat,
        max_lat,

        min_lon,
        max_lon
    )

    print(
        f"\nGrid Points: {len(points)}"
    )

    records = []

    for idx, (lat, lon) in enumerate(
        points,
        start=1
    ):

        print(
            f"[{idx}/{len(points)}]"
        )

        try:

            result = predict_patch(

                lat=lat,

                lon=lon,

                date=date
            )

            if result is None:

                records.append({

                    "lat": lat,
                    "lon": lon,

                    "u": np.nan,
                    "v": np.nan,
                    "speed": np.nan,

                    "u_era5": np.nan,
                    "v_era5": np.nan,
                    "era5_speed": np.nan
                })

                continue

            records.append({

                "lat":
                    result["lat"],

                "lon":
                    result["lon"],

                "u":
                    result["u_final"],

                "v":
                    result["v_final"],

                "speed":
                    result["speed"],

                "u_era5":
                    result["u_era5"],

                "v_era5":
                    result["v_era5"],

                "era5_speed":
                    result["era5_speed"]
            })

        except Exception as e:

            print(
                f"Failed: {e}"
            )

    df = pd.DataFrame(
        records
    )

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    output_csv = (
        "outputs/wind_vectors.csv"
    )

    df.to_csv(

        output_csv,

        index=False
    )

    print(
        f"\nSaved: {output_csv}"
    )

    print(
        f"Vectors: {len(df)}"
    )

    return df


# =====================================
# TEST
# =====================================

if __name__ == "__main__":

    df = predict_scene(

        min_lat=19.00,
        max_lat=19.10,

        min_lon=72.80,
        max_lon=72.90,

        date="2024-01-15"
    )

    print(
        "\nFirst Rows\n"
    )

    print(
        df.head()
    )
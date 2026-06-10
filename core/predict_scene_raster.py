import os
import numpy as np
import pandas as pd

from core.config import (
    GRID_SPACING_KM
)

from core.tile_extractor import (
    TileExtractor
)

from core.feature_extractor import (
    extract_features_batch
)

from core.residual_model import (
    predict_residuals_batch
)

from core.era5_grid import (
    get_era5_grid
)

from core.scene_fetcher import (
    initialize_gee
)




print(
    "\nGRID_SPACING_KM:",
    GRID_SPACING_KM
)

# =====================================
# GRID
# =====================================

def generate_grid(
    min_lat,
    max_lat,
    min_lon,
    max_lon
):

    step_deg = (
        GRID_SPACING_KM / 111.0
    )

    points = []

    lats = np.arange(
        min_lat,
        max_lat + 1e-8,
        step_deg
    )

    lons = np.arange(
        min_lon,
        max_lon + 1e-8,
        step_deg
    )

    for lat in lats:

        for lon in lons:

            points.append(
                (
                    float(lat),
                    float(lon)
                )
            )

    return points


# =====================================
# PREDICT
# =====================================

def predict_scene_raster(

    min_lat,
    max_lat,

    min_lon,
    max_lon,

    date
):

    initialize_gee()

    print("\n====================")
    print("PREDICTION SETTINGS")
    print("====================")

    print(
        "Min Lat:",
        min_lat
    )

    print(
        "Max Lat:",
        max_lat
    )

    print(
        "Min Lon:",
        min_lon
    )

    print(
        "Max Lon:",
        max_lon
    )

    print(
        "\nGRID_SPACING_KM:",
        GRID_SPACING_KM
    )

    # -------------------------
    # GRID
    # -------------------------

    points = generate_grid(

        min_lat,
        max_lat,

        min_lon,
        max_lon
    )

    print(
        f"\nGrid Points: {len(points)}"
    )

    for p in points:
        print(p)

    # -------------------------
    # ERA5
    # -------------------------

    era5_df = get_era5_grid(

        points,
        date
    )

    print(
        f"ERA5 rows: {len(era5_df)}"
    )

    # -------------------------
    # SAR
    # -------------------------

    extractor = TileExtractor(
        "outputs/sar_scene.tif"
    )

    vv_patches = []
    angle_patches = []

    lats = []
    lons = []

    u_era5_list = []
    v_era5_list = []

    # -------------------------
    # PATCH EXTRACTION
    # -------------------------

    for _, row in era5_df.iterrows():

        lat = row["lat"]
        lon = row["lon"]

        patch = extractor.extract_patch(
            lat,
            lon
        )

        if patch is None:

            print(
                "Rejected:",
                lat,
                lon
            )

            continue

        vv_patches.append(
            patch["vv"]
        )

        angle_patches.append(
            patch["angle"]
        )

        lats.append(
            lat
        )

        lons.append(
            lon 
        )

        u_era5_list.append(
            row["u"]
        )

        v_era5_list.append(
            row["v"]
        )
    extractor.close()

    print(
        f"Valid patches: {len(vv_patches)}"
    )

    # -------------------------
    # FEATURES
    # -------------------------

    features = extract_features_batch(

        vv_patches,
        angle_patches
    )

    print(
        "Feature shape:",
        features.shape
    )

    # -------------------------
    # RESIDUALS
    # -------------------------

    residuals = predict_residuals_batch(

        features=features,

        u_era5=np.array(
            u_era5_list
        ),

        v_era5=np.array(
            v_era5_list
        ),

        lat=np.array(
            lats
        ),

        lon=np.array(
            lons
        )
    )

    # -------------------------
    # FINAL WINDS
    # -------------------------

    u_final = (

        np.array(
            u_era5_list
        )

        +

        residuals[
            "residual_u"
        ]
    )

    v_final = (

        np.array(
            v_era5_list
        )

        +

        residuals[
            "residual_v"
        ]
    )

    speed = np.sqrt(

        u_final**2

        +

        v_final**2
    )

    # -------------------------
    # SAVE
    # -------------------------

    df = pd.DataFrame({

        "lat":
            lats,

        "lon":
            lons,

        "u":
            u_final,

        "v":
            v_final,

        "speed":
            speed,

        "u_era5":
            u_era5_list,

        "v_era5":
            v_era5_list
    })

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    output_csv = (
        "outputs/wind_vectors_v3.csv"
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

# =====================================
# TEST
# =====================================

if __name__ == "__main__":

    df = predict_scene_raster(

        min_lat=19.0,
        max_lat=19.1,

        min_lon=72.8,
        max_lon=72.9,

        date="2024-01-15"
    )

    print(
        "\nHead\n"
    )

    print(
        df.head()
    )

    
        
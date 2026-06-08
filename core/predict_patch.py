import numpy as np

from core.data_fetcher import (
    initialize_gee,
    fetch_data
)

from core.feature_extractor import (
    extract_features
)

from core.residual_model import (
    predict_residuals
)

# =====================================
# PREDICT SINGLE PATCH
# =====================================

def predict_patch(
    lat,
    lon,
    date
):
    """
    Predict wind vector for one
    2 km × 2 km SAR patch.

    Returns
    -------
    dict
    """

    # -------------------------
    # Fetch data
    # -------------------------

    data = fetch_data(

        lat=lat,

        lon=lon,

        date=date
    )

    if data is None:

        return None

    # -------------------------
    # Extract features
    # -------------------------

    features = extract_features(

        data["sar_image"],

        data["patch_geometry"]
    )

    # -------------------------
    # Predict residuals
    # -------------------------

    residuals = predict_residuals(

        features_512=features,

        u_era5=data["u_era5"],

        v_era5=data["v_era5"],

        lat=data["lat"],

        lon=data["lon"]
    )

    # -------------------------
    # Final wind
    # -------------------------

    u_final = (

        data["u_era5"]

        +

        residuals["residual_u"]
    )

    v_final = (

        data["v_era5"]

        +

        residuals["residual_v"]
    )

    speed = np.sqrt(

        u_final**2

        +

        v_final**2
    )

    era5_speed = np.sqrt(

        data["u_era5"]**2

        +

        data["v_era5"]**2
    )

    return {

        "date":
            date,

        "lat":
            lat,

        "lon":
            lon,

        "u_era5":
            data["u_era5"],

        "v_era5":
            data["v_era5"],

        "residual_u":
            residuals["residual_u"],

        "residual_v":
            residuals["residual_v"],

        "u_final":
            float(u_final),

        "v_final":
            float(v_final),

        "speed":
            float(speed),

        "era5_speed":
            float(era5_speed)
    }


# =====================================
# TEST
# =====================================

if __name__ == "__main__":

    initialize_gee()

    result = predict_patch(

        lat=19.0760,

        lon=72.8777,

        date="2024-01-15"
    )

    print("\nRESULT\n")

    for k, v in result.items():

        print(
            f"{k}: {v}"
        )
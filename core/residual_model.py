import joblib
import numpy as np
import pandas as pd

from core.config import (
    MODEL_U_PATH,
    MODEL_V_PATH,
    PCA_PATH,
    SCALER_PATH
)

# =====================================
# LOAD ONCE
# =====================================

print("Loading models...")

scaler = joblib.load(
    SCALER_PATH
)

pca = joblib.load(
    PCA_PATH
)

model_u = joblib.load(
    MODEL_U_PATH
)

model_v = joblib.load(
    MODEL_V_PATH
)

print("Models loaded")

# =====================================
# PREDICT RESIDUALS
# =====================================

def predict_residuals(
    features_512,
    u_era5,
    v_era5,
    lat,
    lon
):
    """
    Parameters
    ----------
    features_512 : np.ndarray
        Shape (512,)

    Returns
    -------
    dict
    """

    # -------------------------
    # Shape -> (1,512)
    # -------------------------

    feature_names = [
        f"f_{i:03d}"
        for i in range(512)
    ]

    x = pd.DataFrame(
        [features_512],
        columns=feature_names
    )

    # -------------------------
    # Scaler
    # -------------------------

    x = scaler.transform(
        x
    )

    # -------------------------
    # PCA
    # -------------------------

    x = pca.transform(
        x
    )

    # -------------------------
    # Add metadata
    # -------------------------

    extra = np.array([
        [
            u_era5,
            v_era5,
            lat,
            lon
        ]
    ])

    x = np.hstack(
        [
            x,
            extra
        ]
    )

    # -------------------------
    # Predict
    # -------------------------

    residual_u = float(
        model_u.predict(x)[0]
    )

    residual_v = float(
        model_v.predict(x)[0]
    )

    return {

        "residual_u":
            residual_u,

        "residual_v":
            residual_v
    }


def predict_residuals_batch(

    features,

    u_era5,
    v_era5,

    lat,
    lon
):

    feature_names = [

        f"f_{i:03d}"
        for i in range(512)

    ]

    X = pd.DataFrame(

        features,

        columns=feature_names
    )

    X = scaler.transform(
        X
    )

    X = pca.transform(
        X
    )

    extra = np.column_stack([

        u_era5,
        v_era5,

        lat,
        lon
    ])

    X = np.hstack([
        X,
        extra
    ])

    residual_u = model_u.predict(
        X
    )

    residual_v = model_v.predict(
        X
    )

    return {

        "residual_u":
            residual_u,

        "residual_v":
            residual_v
    }

# =====================================
# TEST
# =====================================

if __name__ == "__main__":

    dummy = np.random.rand(
        5,512
    )

    result = predict_residuals_batch(

        features=dummy,

        u_era5=np.ones(5),

        v_era5=np.ones(5),

        lat=np.ones(5)*19,

        lon=np.ones(5)*72.8
    )

    print(
        result["residual_u"].shape
    )

    print(
        result["residual_v"].shape
    )
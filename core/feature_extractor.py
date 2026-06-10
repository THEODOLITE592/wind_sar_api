import os
import tempfile

import ee
import geemap
import rasterio

import numpy as np

import torch
import torch.nn as nn

from PIL import Image

from torchvision import transforms

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

from core.config import (
    IMAGE_SIZE
)

# =====================================
# DEVICE
# =====================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"Feature Extractor Device: {DEVICE}"
)

# =====================================
# RESNET18
# =====================================

weights = ResNet18_Weights.DEFAULT

resnet = resnet18(
    weights=weights
)

resnet = nn.Sequential(
    *list(resnet.children())[:-1]
)

resnet.eval()

resnet.to(
    DEVICE
)

# =====================================
# TRANSFORM
# =====================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])

# =====================================
# PREPROCESS SAR
# =====================================

def preprocess_sar(
    tif_path
):
    """
    Same preprocessing used
    during training.
    """

    with rasterio.open(
        tif_path
    ) as src:

        vv = src.read(1)

        angle = src.read(2)

    vv = np.nan_to_num(vv)

    angle = np.nan_to_num(angle)

    # -------------------------
    # VV normalization
    # -------------------------

    p2 = np.percentile(
        vv,
        2
    )

    p98 = np.percentile(
        vv,
        98
    )

    vv = np.clip(
        vv,
        p2,
        p98
    )

    vv = (
        vv - vv.min()
    ) / (
        vv.max()
        - vv.min()
        + 1e-8
    )

    # -------------------------
    # Angle normalization
    # -------------------------

    angle = (
        angle
        - angle.min()
    ) / (
        angle.max()
        - angle.min()
        + 1e-8
    )

    # -------------------------
    # Build RGB image
    # -------------------------

    img = np.stack(

        [
            vv,
            angle,
            vv
        ],

        axis=-1
    )

    img = (
        img * 255
    ).astype(
        np.uint8
    )

    img = Image.fromarray(
        img
    )

    return transform(
        img
    )

def preprocess_arrays(
    vv,
    angle
):

    vv = np.nan_to_num(vv)
    angle = np.nan_to_num(angle)

    p2 = np.percentile(vv, 2)
    p98 = np.percentile(vv, 98)

    vv = np.clip(
        vv,
        p2,
        p98
    )

    vv = (
        vv - vv.min()
    ) / (
        vv.max() - vv.min() + 1e-8
    )

    angle = (
        angle - angle.min()
    ) / (
        angle.max() - angle.min() + 1e-8
    )

    img = np.stack(
        [
            vv,
            angle,
            vv
        ],
        axis=-1
    )

    img = (
        img * 255
    ).astype(
        np.uint8
    )

    img = Image.fromarray(
        img
    )

    return transform(
        img
    )

def extract_features_array(
    vv,
    angle
):

    tensor = preprocess_arrays(
        vv,
        angle
    )

    tensor = (
        tensor
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        features = resnet(
            tensor
        )

    features = (
        features
        .cpu()
        .numpy()
        .flatten()
    )

    return features

# =====================================
# EXPORT EE IMAGE
# =====================================

def export_patch(
    sar_image,
    patch_geometry
):
    """
    Export temporary SAR patch
    from Earth Engine.
    """

    tmp_file = tempfile.NamedTemporaryFile(

        suffix=".tif",

        delete=False
    )

    tmp_path = tmp_file.name

    tmp_file.close()

    geemap.ee_export_image(

        sar_image,

        filename=tmp_path,

        scale=10,

        region=patch_geometry,

        file_per_band=False
    )

    return tmp_path


# =====================================
# FEATURE EXTRACTION
# =====================================

def extract_features_batch(
    vv_patches,
    angle_patches
):

    tensors = []

    for vv, angle in zip(
        vv_patches,
        angle_patches
    ):

        tensor = preprocess_arrays(
            vv,
            angle
        )

        tensors.append(
            tensor
        )

    batch = torch.stack(
        tensors
    ).to(
        DEVICE
    )

    with torch.no_grad():

        features = resnet(
            batch
        )

    features = (
        features
        .cpu()
        .numpy()
    )

    features = features.reshape(
        features.shape[0],
        -1
    )

    return features

def extract_features(
    sar_image,
    patch_geometry
):
    """
    Returns
    -------
    np.ndarray

    Shape:
    (512,)
    """

    tif_path = export_patch(

        sar_image,

        patch_geometry
    )

    try:

        tensor = preprocess_sar(
            tif_path
        )

        tensor = (
            tensor
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():

            features = resnet(
                tensor
            )

        features = (

            features

            .cpu()

            .numpy()

            .flatten()
        )

        return features

    finally:

        if os.path.exists(
            tif_path
        ):
            os.remove(
                tif_path
            )


# =====================================
# TEST
# =====================================

if __name__ == "__main__":

    from core.data_fetcher import (
        initialize_gee,
        fetch_data
    )

    initialize_gee()

    result = fetch_data(

        lat=19.0760,

        lon=72.8777,

        date="2024-01-15"
    )

    if result:

        feats = extract_features(

            result["sar_image"],

            result["patch_geometry"]
        )

        print(
            "Feature Shape:",
            feats.shape
        )

        print(
            feats[:10]
        )

if __name__ == "__main__":

    from core.tile_extractor import TileExtractor

    extractor = TileExtractor(
        "outputs/sar_scene.tif"
    )

    vv_patches = []
    angle_patches = []

    locations = [

        (19.02, 72.82),
        (19.04, 72.84),
        (19.06, 72.86)

    ]

    for lat, lon in locations:

        patch = extractor.extract_patch(
            lat,
            lon
        )

        if patch is None:
            continue

        vv_patches.append(
            patch["vv"]
        )

        angle_patches.append(
            patch["angle"]
        )

    features = extract_features_batch(

        vv_patches,
        angle_patches
    )

    print(
        features.shape
    )

    extractor.close()
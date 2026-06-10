import ee
import numpy as np

from core.scene_fetcher import (
    initialize_gee,
    fetch_scene
)

from core.feature_extractor import (
    extract_features,
    extract_features_array
)

from core.tile_extractor import (
    TileExtractor
)

# =====================================
# CONFIG
# =====================================

LAT = 19.05
LON = 72.85

DATE = "2024-01-15"

# =====================================
# GEE FEATURE VECTOR
# =====================================

initialize_gee()

scene_data = fetch_scene(

    min_lat=19.0,
    max_lat=19.1,

    min_lon=72.8,
    max_lon=72.9,

    date=DATE
)

scene = scene_data["scene"]

patch = (

    ee.Geometry.Point(
        [LON, LAT]
    )

    .buffer(
        1000
    )

    .bounds()
)

print("\nExtracting GEE features...")

f1 = extract_features(
    scene,
    patch
)

print(
    "f1 shape:",
    f1.shape
)

# =====================================
# LOCAL FEATURE VECTOR
# =====================================

print("\nExtracting local features...")

extractor = TileExtractor(
    "outputs/sar_scene.tif"
)

tile = extractor.extract_patch(
    lat=LAT,
    lon=LON
)

f2 = extract_features_array(

    tile["vv"],
    tile["angle"]
)

print(
    "f2 shape:",
    f2.shape
)

extractor.close()

# =====================================
# COMPARISON
# =====================================

diff = np.linalg.norm(
    f1 - f2
)

mean_abs = np.mean(
    np.abs(f1 - f2)
)

max_abs = np.max(
    np.abs(f1 - f2)
)

corr = np.corrcoef(
    f1,
    f2
)[0, 1]

print("\n===================")
print("FEATURE COMPARISON")
print("===================")

print(
    "L2 Distance:",
    diff
)

print(
    "Mean Abs Difference:",
    mean_abs
)

print(
    "Max Abs Difference:",
    max_abs
)

print(
    "Correlation:",
    corr
)
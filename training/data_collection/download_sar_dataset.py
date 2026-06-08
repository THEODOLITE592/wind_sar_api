import ee
import geemap
import pandas as pd
import random
import os
import math

# ====================================================
# CONFIG
# ====================================================

PROJECT_ID = "crop-472500"

TARGET_SAMPLES = 500

PATCH_SIZE_METERS = 2000

OUTPUT_DIR = "data/sar_images"

# Coastal regions
REGIONS = [
    ("Kutch", [68.5, 22.0, 70.5, 23.5]),
    ("Khambhat", [71.0, 21.0, 73.0, 22.5]),
    ("Mumbai", [72.5, 18.5, 73.3, 19.5]),
    ("Goa", [73.5, 15.0, 74.3, 15.8]),
    ("Kerala", [75.0, 9.0, 76.5, 11.0]),
]

# ====================================================
# INIT
# ====================================================

ee.Initialize(project=PROJECT_ID)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

records = []

sample_idx = 0

used_points = []

# ====================================================
# HELPER
# ====================================================

def point_too_close(lat, lon, threshold=0.01):
    """
    ~1 km separation
    """

    for plat, plon in used_points:

        d = math.sqrt(
            (lat - plat) ** 2 +
            (lon - plon) ** 2
        )

        if d < threshold:
            return True

    return False

# ====================================================
# WATER MASK
# ====================================================

water_mask = ee.Image(
    "JRC/GSW1_4/GlobalSurfaceWater"
).select("occurrence")

# ====================================================
# MAIN LOOP
# ====================================================

while sample_idx < TARGET_SAMPLES:

    region_name, bbox = random.choice(REGIONS)

    min_lon, min_lat, max_lon, max_lat = bbox

    lon = random.uniform(min_lon, max_lon)
    lat = random.uniform(min_lat, max_lat)

    if point_too_close(lat, lon):
        continue

    point = ee.Geometry.Point([lon, lat])

    patch = (
        point
        .buffer(PATCH_SIZE_METERS / 2)
        .bounds()
    )

    try:

        # ------------------------------------
        # Water coverage
        # ------------------------------------

        water_fraction = (
            water_mask
            .gt(50)
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=patch,
                scale=30,
                maxPixels=1e8
            )
            .get("occurrence")
            .getInfo()
        )

        if water_fraction is None:
            continue

        if water_fraction < 0.80:
            continue

        # ------------------------------------
        # Sentinel image
        # ------------------------------------

        collection = (
            ee.ImageCollection(
                "COPERNICUS/S1_GRD"
            )
            .filterBounds(point)
            .filterDate(
                "2023-01-01",
                "2024-12-31"
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

        count = collection.size().getInfo()

        if count == 0:
            continue

        image = ee.Image(
            collection.first()
        )

        info = image.getInfo()

        scene_id = info["id"]

        timestamp = (
            info["properties"]
            ["system:time_start"]
        )

        sample_name = (
            f"sample_{sample_idx:05d}"
        )

        filename = (
            f"{OUTPUT_DIR}/"
            f"{sample_name}.tif"
        )

        print(
            f"[{sample_idx+1}/{TARGET_SAMPLES}]",
            sample_name,
            region_name
        )

        geemap.ee_export_image(
            image.clip(patch),
            filename=filename,
            scale=10,
            region=patch,
            file_per_band=False
        )

        used_points.append(
            (lat, lon)
        )

        records.append({

            "sample_id":
                sample_name,

            "region":
                region_name,

            "scene_id":
                scene_id,

            "lat":
                lat,

            "lon":
                lon,

            "timestamp":
                timestamp,

            "file_path":
                filename,

            "u_cmod":
                None,

            "v_cmod":
                None,

            "u_ascat":
                None,

            "v_ascat":
                None,

            "residual_u":
                None,

            "residual_v":
                None
        })

        sample_idx += 1

    except Exception as e:

        print("Skip:", e)

# ====================================================
# SAVE CSV
# ====================================================

df = pd.DataFrame(records)

df.to_csv(
    "data/metadata.csv",
    index=False
)

print("\n================================")
print(
    f"Generated {len(df)} samples"
)
print(
    "Metadata saved"
)
print("================================")
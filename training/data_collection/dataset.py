import ee
import geemap
import pandas as pd
import random
import os
import math
from datetime import timedelta

# =====================================================
# CONFIG
# =====================================================

PROJECT_ID = "crop-472500"

PATCH_SIZE_METERS = 2000

OUTPUT_DIR = "new_data/sar_images"

TARGET_PER_REGION = 300

REGIONS = {

    "Kutch": [68.5, 22.0, 70.5, 23.5],

    "Khambhat": [71.0, 21.0, 73.0, 22.5],

    "Mumbai": [72.5, 18.5, 73.3, 19.5],

    "Goa": [73.5, 15.0, 74.3, 15.8],

    "Kerala": [75.0, 9.0, 76.5, 11.0]
}

TOTAL_TARGET = (
    len(REGIONS)
    * TARGET_PER_REGION
)

# =====================================================
# INIT
# =====================================================

ee.Initialize(project=PROJECT_ID)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

records = []

used_points = []

sample_idx = 0

region_counts = {
    region: 0
    for region in REGIONS
}

# =====================================================
# ERA5
# =====================================================

ERA5 = ee.ImageCollection(
    "ECMWF/ERA5/HOURLY"
)

# =====================================================
# HELPERS
# =====================================================

def point_too_close(
    lat,
    lon,
    threshold=0.03
):
    """
    ~3 km separation
    """

    for plat, plon in used_points:

        d = math.sqrt(
            (lat - plat) ** 2 +
            (lon - plon) ** 2
        )

        if d < threshold:
            return True

    return False


def get_era5_uv(
    lat,
    lon,
    timestamp_ms
):

    point = ee.Geometry.Point(
        [lon, lat]
    )

    ts = pd.to_datetime(
        timestamp_ms,
        unit="ms"
    )

    start = (
        ts - timedelta(hours=1)
    )

    end = (
        ts + timedelta(hours=1)
    )

    image = (

        ERA5

        .filterDate(
            start.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
            end.strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
        )

        .sort(
            "system:time_start"
        )

        .first()
    )

    if image is None:
        return None, None

    values = image.reduceRegion(

        reducer=ee.Reducer.first(),

        geometry=point,

        scale=25000

    )

    values_dict = values.getInfo()

    if values_dict is None:
        return None, None

    u10 = values_dict.get(
        "u_component_of_wind_10m"
    )

    v10 = values_dict.get(
        "v_component_of_wind_10m"
    )

    return u10, v10


# =====================================================
# WATER MASK
# =====================================================

water_mask = (

    ee.Image(
        "JRC/GSW1_4/GlobalSurfaceWater"
    )

    .select("occurrence")

)

# =====================================================
# MAIN LOOP
# =====================================================

while sample_idx < TOTAL_TARGET:

    remaining_regions = [

        r

        for r in REGIONS

        if region_counts[r]
        < TARGET_PER_REGION

    ]

    region_name = random.choice(
        remaining_regions
    )

    bbox = REGIONS[region_name]

    min_lon, min_lat, max_lon, max_lat = bbox

    lon = random.uniform(
        min_lon,
        max_lon
    )

    lat = random.uniform(
        min_lat,
        max_lat
    )

    if point_too_close(
        lat,
        lon
    ):
        continue

    point = ee.Geometry.Point(
        [lon, lat]
    )

    patch = (

        point

        .buffer(
            PATCH_SIZE_METERS / 2
        )

        .bounds()

    )

    try:

        # =================================
        # WATER CHECK
        # =================================

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

        # =================================
        # SENTINEL-1
        # =================================

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

        random_idx = random.randint(
            0,
            count - 1
        )

        image = ee.Image(

            collection
            .toList(count)
            .get(random_idx)

        )

        info = image.getInfo()

        scene_id = info["id"]

        timestamp = (
            info["properties"]
            ["system:time_start"]
        )

        # =================================
        # ERA5
        # =================================

        u10, v10 = get_era5_uv(
            lat,
            lon,
            timestamp
        )

        if u10 is None or v10 is None:
            continue

        speed_era5 = (
            (u10 ** 2 + v10 ** 2)
            ** 0.5
        )

        # =================================
        # EXPORT
        # =================================

        sample_name = (
            f"sample_{sample_idx:05d}"
        )

        filename = (

            f"{OUTPUT_DIR}/"
            f"{sample_name}.tif"

        )

        print(
            f"[{sample_idx+1}/{TOTAL_TARGET}]",
            region_name,
            f"({region_counts[region_name]+1}/{TARGET_PER_REGION})",
            f"ERA5={speed_era5:.2f}m/s"
        )

        geemap.ee_export_image(

            image.clip(patch),

            filename=filename,

            scale=10,

            region=patch,

            file_per_band=False

        )

        # =================================
        # SAVE ROW
        # =================================

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

            "file":
                filename,

            "u_era5":
                u10,

            "v_era5":
                v10,

            "speed_era5":
                speed_era5
        })

        used_points.append(
            (lat, lon)
        )

        region_counts[
            region_name
        ] += 1

        sample_idx += 1

        # checkpoint

        if sample_idx % 25 == 0:

            pd.DataFrame(
                records
            ).to_csv(
                "new_data/metadata.csv",
                index=False
            )

            print(
                "\nCheckpoint saved\n"
            )

    except Exception as e:

        print(
            "Skip:",
            e
        )

# =====================================================
# FINAL SAVE
# =====================================================

df = pd.DataFrame(records)

df.to_csv(
    "new_data/metadata.csv",
    index=False
)

print("\n=================================")
print("DATA COLLECTION COMPLETE")
print("=================================")

print(
    f"Total Samples: {len(df)}"
)

print("\nRegion Counts:")

for region in region_counts:

    print(
        region,
        region_counts[region]
    )

print(
    "\nSaved: data/metadata.csv"
)
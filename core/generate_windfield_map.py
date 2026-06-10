import os

import numpy as np
import pandas as pd

import rasterio

import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from pyproj import Transformer


# =====================================
# GENERATE WIND FIELD MAP
# =====================================

def generate_windfield_map(

    csv_path="outputs/wind_vectors_v3.csv",

    sar_path="outputs/sar_scene.tif",

    output_path="outputs/wind_field.png"
):

    # -------------------------
    # LOAD WIND VECTORS
    # -------------------------

    df = pd.read_csv(
        csv_path
    )

    u = df["u"].values
    v = df["v"].values

    speed = df["speed"].values

    # -------------------------
    # LOAD SAR
    # -------------------------

    with rasterio.open(
        sar_path
    ) as src:

        vv = src.read(1)

        bounds = src.bounds

        raster_crs = src.crs

        raster_transform = src.transform

        width = src.width

        height = src.height

    # -------------------------
    # CONVERT LAT/LON -> UTM
    # -------------------------

    transformer = Transformer.from_crs(

        "EPSG:4326",

        raster_crs,

        always_xy=True
    )

    x, y = transformer.transform(

        df["lon"].values,

        df["lat"].values
    )

    # -------------------------
    # RASTER CORNERS -> LAT/LON
    # -------------------------
    inverse_transformer = Transformer.from_crs(
        raster_crs,
        "EPSG:4326",
        always_xy=True
    )

    lon_min, lat_min = inverse_transformer.transform(
        bounds.left,
        bounds.bottom
    )

    lon_max, lat_max = inverse_transformer.transform(
        bounds.right,
        bounds.top
    )


    # -------------------------
    # DEBUG
    # -------------------------

    print("\nRaster CRS:")
    print(raster_crs)

    print("\nRaster Bounds:")
    print(bounds)

    print("\nVector X Range:")
    print(x.min(), x.max())

    print("\nVector Y Range:")
    print(y.min(), y.max())

    # -------------------------
    # INTERPOLATION GRID
    # -------------------------

    grid_x, grid_y = np.meshgrid(

        np.linspace(
            x.min(),
            x.max(),
            300
        ),

        np.linspace(
            y.min(),
            y.max(),
            300
        )
    )

    grid_lon, grid_lat = inverse_transformer.transform(
        grid_x,
        grid_y
    )


    speed_grid = griddata(

        (x, y),

        speed,
 
        (grid_x, grid_y),

        method="cubic"
    )

    u_grid = griddata(

        (x, y),

        u,

        (grid_x, grid_y),

        method="cubic"
    )

    v_grid = griddata(

        (x, y),

        v,

        (grid_x, grid_y),

        method="cubic"
    )

    # -------------------------
    # NORMALIZE VV
    # -------------------------

    vv = np.nan_to_num(
        vv
    )

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
    # PLOT
    # -------------------------

    plt.figure(
        figsize=(12, 10)
    )

    plt.imshow(

        vv,

        cmap="gray",

        alpha = 0.95,

        extent=[

            lon_min,
            lon_max,

            lat_min,
            lat_max
        ],

        origin="upper"
    )

    cf = plt.contourf(

        grid_lon,
        grid_lat,

        speed_grid,

        levels = 50,

        cmap="turbo",

        alpha = 0.25
    )

    plt.quiver(

        grid_lon[::12,::12],
        grid_lat[::12,::12],

        u_grid[::12,::12],
        v_grid[::12,::12],

        color = "black",

        scale = 120,

        width = 0.002
    )



    plt.colorbar(

        cf,

        label="Wind Speed (m/s)"
    )

    plt.title(
        "SAR Wind Field"
    )

    plt.xlabel(
        "Longitude"
    )

    plt.ylabel(
        "Latitude"
    )

    plt.tight_layout()

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    plt.savefig(

        output_path,

        dpi=300,

        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nSaved: {output_path}"
    )

    return output_path


# =====================================
# TEST
# =====================================

if __name__ == "__main__":

    generate_windfield_map()
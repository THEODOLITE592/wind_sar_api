import os

import pandas as pd
import matplotlib.pyplot as plt

# =====================================
# GENERATE WIND FIELD
# =====================================

def generate_windfield(
    csv_path="outputs/wind_vectors.csv",
    output_path="outputs/wind_field.png"
):

    df = pd.read_csv(csv_path)

    if len(df) == 0:

        print("No vectors found")

        return

    lon = df["lon"].values
    lat = df["lat"].values

    u = df["u"].values
    v = df["v"].values

    speed = df["speed"].values

    plt.figure(
        figsize=(10, 8)
    )

    q = plt.quiver(

        lon,
        lat,

        u,
        v,

        speed,

        scale=None
    )

    plt.colorbar(
        q,
        label="Wind Speed (m/s)"
    )

    plt.xlabel("Longitude")

    plt.ylabel("Latitude")

    plt.title(
        "SAR Enhanced Wind Field"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# =====================================
# TEST
# =====================================

if __name__ == "__main__":

    generate_windfield()
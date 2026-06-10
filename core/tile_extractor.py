import rasterio

import numpy as np

from pyproj import Transformer

from core.config import (
    PATCH_SIZE_M
)

# =====================================
# LOAD SAR SCENE
# =====================================

class TileExtractor:

    def __init__(self, tif_path):

        self.src = rasterio.open(
            tif_path
        )

        self.vv = self.src.read(1)

        self.angle = self.src.read(2)

        self.transform = self.src.transform

        self.height = self.src.height
        self.width = self.src.width

        self.crs = self.src.crs

        self.ll_to_utm = Transformer.from_crs(
            "EPSG:4326",
            self.crs,
            always_xy=True
        )

        # pixel size in meters

        self.pixel_size = abs(
            self.transform.a
        )

        self.patch_pixels = int(
            PATCH_SIZE_M /
            self.pixel_size
        )

        self.half_patch = (
            self.patch_pixels // 2
        )

        print(
            f"Scene size: "
            f"{self.width} x {self.height}"
        )

        print(
            f"Pixel size: "
            f"{self.pixel_size:.2f} m"
        )

        print(
            f"Patch size: "
            f"{self.patch_pixels} pixels"
        )

    # =================================
    # LAT/LON -> PIXEL
    # =================================

    def latlon_to_pixel(
        self,
        lat,
        lon
    ):

        x_utm, y_utm = (
            self.ll_to_utm.transform(
                lon,
                lat
            )
        )

        row, col = self.src.index(
            x_utm,
            y_utm
        )

        return row, col

    # =================================
    # EXTRACT PATCH
    # =================================

    def extract_patch(
        self,
        lat,
        lon
    ):

        row, col = self.latlon_to_pixel(
            lat,
            lon
        )

        r0 = row - self.half_patch
        r1 = row + self.half_patch

        c0 = col - self.half_patch
        c1 = col + self.half_patch

        print(
            "\nPoint:",
            lat,
            lon
            )
        
        print(
            "Pixel:",
            row,
            col
            )
        
        print(
            "Window:",
            r0,
            r1,
            c0,
            c1
            )
        
        print(
            "Image:",
            self.height,
            self.width
            )

        # skip edges

        if (
            r0 < 0 or
            c0 < 0 or
            r1 >= self.height or
            c1 >= self.width
        ):
            print(
                "REJECTED"
            )
            return None

        vv_patch = self.vv[
            r0:r1,
            c0:c1
        ]

        angle_patch = self.angle[
            r0:r1,
            c0:c1
        ]

        return {

            "vv": vv_patch,

            "angle": angle_patch,

            "row": row,

            "col": col
        }

    # =================================
    # CLOSE
    # =================================

    def close(self):

        self.src.close()

if __name__ == "__main__":

    extractor = TileExtractor(
        "outputs/sar_scene.tif"
    )

    print(
        "\nRaster Bounds:"
    )

    print(
        extractor.src.bounds
    )

    test_points = [

        (19.05, 72.85),      # corner
        (19.025, 72.825),    # center
        (19.02, 72.82),
        (19.04, 72.84)

    ]

    for lat, lon in test_points:

        print(
            "\n===================="
        )

        patch = extractor.extract_patch(
            lat,
            lon
        )

        if patch is None:

            print(
                "Patch outside scene"
            )

        else:

            print(
                "Patch shape VV:",
                patch["vv"].shape
            )

            print(
                "Patch shape Angle:",
                patch["angle"].shape
            )

    extractor.close()
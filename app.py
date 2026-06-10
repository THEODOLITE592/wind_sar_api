import streamlit as st

from core.download_scene import (
    download_scene
)

from core.predict_scene_raster import (
    predict_scene_raster
)

from core.generate_windfield_map import (
    generate_windfield_map
)

st.set_page_config(
    page_title="SAR Wind Retrieval",
    layout="wide"
)

st.title(
    "🌊 SAR Wind Field Generator"
)

st.subheader(
    "Generate coastal wind field from Sentinel-1 SAR"
)

# -------------------------
# INPUTS
# -------------------------

col1, col2 = st.columns(2)

with col1:

    min_lat = st.number_input(
        "Min Latitude",
        value=19.0
    )

    max_lat = st.number_input(
        "Max Latitude",
        value=19.1
    )

with col2:

    min_lon = st.number_input(
        "Min Longitude",
        value=72.8
    )

    max_lon = st.number_input(
        "Max Longitude",
        value=72.9
    )

date = st.date_input(
    "Date"
)

# -------------------------
# BUTTON
# -------------------------

if st.button(
    "Generate Wind Field"
):

    st.info(
        "Starting pipeline..."
    )

    # STEP 1
    with st.spinner(
        "Downloading Sentinel-1 scene..."
    ):

        download_scene(

            min_lat=min_lat,
            max_lat=max_lat,

            min_lon=min_lon,
            max_lon=max_lon,

            date=str(date)
        )

    # STEP 2
    with st.spinner(
        "Predicting winds..."
    ):

        predict_scene_raster(

            min_lat=min_lat,
            max_lat=max_lat,

            min_lon=min_lon,
            max_lon=max_lon,

            date=str(date)
        )

    # STEP 3
    with st.spinner(
        "Generating wind field map..."
    ):

        generate_windfield_map()

    st.success(
    "Completed"
)

st.image(
    "outputs/wind_field.png",
    width=700
)

import pandas as pd

df = pd.read_csv(
    "outputs/wind_vectors_v3.csv"
)

st.subheader(
    "Wind Statistics"
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "Vectors",
    len(df)
)

c2.metric(
    "Mean Speed (m/s)",
    round(
        df["speed"].mean(),
        2
    )
)

c3.metric(
    "Max Speed (m/s)",
    round(
        df["speed"].max(),
        2
    )
)

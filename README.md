# SAR Wind Retrieval from Sentinel-1 SAR

A machine learning pipeline for retrieving coastal wind fields from Sentinel-1 SAR imagery.

## Features

* Download Sentinel-1 SAR scenes using Google Earth Engine
* Extract VV backscatter and incidence angle information
* Generate ERA5 wind field priors
* Extract deep features using ResNet18
* Predict wind corrections using XGBoost residual models
* Generate wind vector maps and wind speed visualizations
* Interactive Streamlit web application

## Project Structure

```text
app.py

core/
├── config.py
├── download_scene.py
├── scene_fetcher.py
├── era5_grid.py
├── tile_extractor.py
├── feature_extractor.py
├── residual_model.py
├── predict_scene_raster.py
├── generate_windfield_map.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Run Streamlit App

```bash
streamlit run app.py
```

## Workflow

1. User selects Area of Interest (AOI) and date.
2. Sentinel-1 SAR scene is downloaded.
3. ERA5 wind data is sampled over the AOI.
4. SAR patches are extracted.
5. ResNet18 extracts deep SAR features.
6. XGBoost residual models estimate wind corrections.
7. Wind vectors and wind speed maps are generated.
8. Results are displayed in the Streamlit interface.

## Technologies Used

* Python
* Streamlit
* PyTorch
* XGBoost
* Google Earth Engine
* Rasterio
* NumPy
* Pandas
* Matplotlib
* SciPy

## Author

Yash Babu

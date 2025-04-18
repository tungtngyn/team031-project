VISUALIZATION, EXPLORATION, AND ANALYSIS OF AIRFARE PRICES
CSE 6242 - Final Project
Team 31 - Spring 2025

Members:
* Tung Nguyen (tnguyen844)
* Narciso Faustino (nfaustino3)
* Yangxi Li (yli3725)
* Terri Tsai (ttsai43)
* Samuel Au (sau31)



DESCRIPTION
- This webapp is built in Python using `bokeh` and `holoviews` (for interactive visuals) and `scikit-learn`, `xgboost`, `catboost`, and `prophet` (for analytics).

DATA
- Download the csv file from [Kaggle] https://www.kaggle.com/datasets/bhavikjikadara/us-airline-flight-routes-and-fares-1993-2024/data
  - Note you will need a create a Kaggle account to downloads
  - Downloaded csv file is saved at `./data/raw-data.csv`

DATA CLEANING & TRANSFORMATIONS
- Open `./dev/data_preprocess.ipynb` and run Juypter Notebook
  - Creates processed data `./data/processed-data.csv` which is used by webapp

GENERATE ANALYTICAL MODELS TO LOAD INTO WEBAPP
(5 Models: XGBoost, CatBoost, Decision Tree, Random Forest, FB Prophet)
  - TODO !!! 
  - ADD how to get outputs saved from each ML model

ENIVRONMENT SETUP & INSTALLATION
- Install Python 3.12 from the [python website](https://www.python.org/downloads/release/python-3120/), or using a package manager (e.g., `homebrew` or `anaconda`). 

- Create a virtual environment and install the packages in `requirements.txt`. Example commands for macOS and python's built-in `venv` module shown below.

```bash
python3.12 -m venv ./<your-venv-name>

source <your-venv-name>/activate

pip install -r requirements.txt
```



EXECUTION
To run the webapp locally, 

1. Open your terminal app of choice (e.g., powershell, zsh, bash, etc.) and activate the virtual environment you created in the last step.

2. Navigate to the top-level folder where `app.py` is. 

3. Run the following command:

```bash
bokeh serve --show app.py
```

The app should now be running on `localhost:5006/app`. The `--show` command should automatically open a browser window.
Note: should take a few seconds to boot up depending on your system hardware specfications.

WEB APPLICATION LAYOUT & USER INTERACTION
- Layout
  - Top section
    - Include 3 dropdowns that are inputs for the visualizations and ML models
      - Origin Airport, Destination Airport, and Season of Travel
  - Center-left section
    - Choropleth Map of All Airfares from Origin Airport (Top-Left)
    - Time Series Line Graph of Airfare (Bottom-Left)
  - Center-right section (Note the Tabs)
    - Analysis by Fare Tab
      - Histogram of Airfare Market Share (Top)
      - Seasonal Box Plots of Airfares (Bottom)
    - Analysis by Airline Tab
      - Bar Chart of Average Airfare of Largest Carrier by Airline (Top)
      - Bar Chart of Average Airfare of Lowest Cost Carrier by Airline (Bottom)
  - Bottom section
      - ML Model Selection dropdowns
        - XGBoost, CatBoost, Decision Tree, Random Forest, FB Prophet
      - Analyze Button
      - Estimated Airfare Price


- Interactivity
  - Graphics that update with 2 inputs: Origin & Destination Airports
    - All graphics
  - Graphics that update with 3 inputs: Origin, Destination, & Season
    - Histogram & Bar Charts
  - ML Models and Analyze Button
    - All models except for Prophet forecasting require all 3 inputs
      - Click "Analyze" to get estimated airfare price
    - Prophet forecasting requires 2 inputs origin & destination
      - Click "Analyze" to update time series graph by appending the 2 year forecast through 2026-Q1
      - Note: Not all flights are eligible for forecasting, so graph will not update
        - Will get print output from console
        - Select another flight route


- Toolbars and Hover Options
  - All graphics have a small toolbar attached to the right side
    - Controls pan, zoom, export, reset, etc.
  - Most graphics have hover options to show more detail

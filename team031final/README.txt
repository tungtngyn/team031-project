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



INSTALLATION
- Install Python 3.12 from the python website (https://www.python.org/downloads/release/python-3120/), or using a package manager (e.g., `homebrew` or `anaconda`). 

- Create a virtual environment and install the packages in `requirements.txt`. Example commands for macOS and python's built-in `venv` module shown below.

```bash
python3.12 -m venv ./<your-venv-name>

source <your-venv-name>/activate

pip install -r requirements.txt
```

- Using this Google Drive link (https://drive.google.com/drive/folders/1gEzsVXkbRj4pyH-g0TAjDDVeKT_BfOaU?usp=sharing),
  
  - Download all the files in /data/ (Google Drive) and place them in ./CODE/data/ (Local Drive)
  
  - Download the model file in /models/ (Google Drive) and place them in ./CODE/models/ (Local Drive)

- If you prefer to rebuild the /data/ files and /models/ files, please follow the instructions in the APPENDIX section of this README.txt



EXECUTION
To run the webapp locally, 

1. Open your terminal app of choice (e.g., powershell, zsh, bash, etc.) and activate the virtual environment you created in the last step.

2. Navigate to the top-level folder where `app.py` is (e.g., ./CODE/). 

3. Run the following command:

```bash
bokeh serve --show app.py
```

The app should now be running on `localhost:5006/app`. The `--show` command should automatically open a browser window.
Note: should take a few seconds to boot up depending on your system hardware specfications.



APPENDIX

Note: 
- The Jupyter Notebooks and Python Scripts were developed independently of the app itself.
- Thus, you may need to update filepaths inside the code itself depending on where you run the script / notebook. 

DATA
- Download the csv file from Kaggle (https://www.kaggle.com/datasets/bhavikjikadara/us-airline-flight-routes-and-fares-1993-2024/data)
  - Note you will need a create a Kaggle account to downloads
  - Downloaded csv file is saved at `./data/raw-data.csv`


DATA CLEANING & TRANSFORMATIONS
- Open `./APPENDIX/data_preprocess.ipynb` and run Juypter Notebook
  - Creates processed data `./data/processed-data.csv` which is used by webapp


GENERATE ANALYTICAL MODELS TO LOAD INTO WEBAPP
(5 Models: Decision Tree, Random Forest, XGBoost, CatBoost, FaceBook Prophet)

- Decision Tree & Random Forest (Tree-Based)
  - Run `./APPENDIX/DecisionTree_RandomForest.ipynb` to export models via pickle
    - Creates `./CODE/models/best_decision_tree_regressor.pkl`
    - Creates `./CODE/models/decision_and_forest_model_columns.pkl`

- XGBoost & CatBoost (Gradient Boosting)
  - Run `./APPENDIX/XGBoost.ipynb` to export model via pickle
    - Creates `./CODE/models/xgb_airfare_model.pkl`
  - Run `./APPENDIX/CatBoost.ipynb` to export model via pickle
    - Creates `./CODE/models/catboost_airfare_model.pkl`

- FB Prophet (Time Series Forecasting)
  - Run `./CODE/src/fbp_tsa.py` using Section III.ii Option A to export forecasted dataframes as JSON
    - Creates `./CODE/models/prophet_model_fare_forecast.json` when using custom function param ycol='fare'
    - Creates `./CODE/models/prophet_model_farelg_forecast.json` when using custom function param ycol='fare_lg'
    - Creates `./CODE/models/prophet_model_farelow_forecast.json` when using custom function param ycol='fare_low'

- Note: All models are loaded into webapp during launch


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
      - Note: Not all flights are eligible for forecasting, so line graph will not update
        - Will get error print output from console
        - Select another flight route

- Toolbars and Hover Options
  - All graphics have a small toolbar attached to the right side
    - Controls pan, zoom, export, reset, etc.
  - Most graphics have hover options to show more detail


TERMINATING WEBAPP
- Close the tab in the browser and quit (CTRL+C or CMD+C) the localhost in the terminal
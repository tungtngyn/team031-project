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
- Install Python 3.12 from the [python website](https://www.python.org/downloads/release/python-3120/), or using a package manager (e.g., `homebrew` or `anaconda`). 

- Create a virtual environment and install the packages in `requirements.txt`. Example commands for macOS and python's built-in `venv` module shown below.

```bash
python3.12 -m venv ./your-venv-name

source your-venv-name/activate

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
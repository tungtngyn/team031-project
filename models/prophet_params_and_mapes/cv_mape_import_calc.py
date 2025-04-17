import json
from statistics import mean

# Average Airfare MAPES using 10-fold CV Parameter Grid Search on PreCOVID Data
# NOTE average airfare is across all carriers
with open(r'./models/prophet_params_and_mapes/prophet_model_fare_mapes_preCOVID.json', 'r') as file_in:
  best_mapes_dict = json.load(file_in)  # best_mapes is dict

mean_mape = 100 * mean(best_mapes_dict.values())
min_mape = 100 * min(best_mapes_dict.values())
max_mape = 100 * max(best_mapes_dict.values())

print(f'Best MAPES from Prophet CV Optimization for {len(best_mapes_dict)} Flights')
print(f'Note: This is for Average Airfare')
print(f' avg MAPE: {mean_mape:.2f}%')
print(f' min MAPE: {min_mape:.2f}% ({min(best_mapes_dict, key=best_mapes_dict.get)})')
print(f' max MAPE: {max_mape:.2f}% ({max(best_mapes_dict, key=best_mapes_dict.get)})')
print()

# Average Airfare MAPES using 10-fold CV Parameter Grid Search on PreCOVID Data
# NOTE average airfare is from the largest carrier
with open(r'./models/prophet_params_and_mapes/prophet_model_farelg_mapes_preCOVID.json', 'r') as file_in:
  best_mapes_dict = json.load(file_in)  # best_mapes is dict

mean_mape = 100 * mean(best_mapes_dict.values())
min_mape = 100 * min(best_mapes_dict.values())
max_mape = 100 * max(best_mapes_dict.values())

print(f'Best MAPES from Prophet CV Optimization for {len(best_mapes_dict)} Flights')
print(f'Note: This is for Average Airfare of the Largest Carrier')
print(f' avg MAPE: {mean_mape:.2f}%')
print(f' min MAPE: {min_mape:.2f}% ({min(best_mapes_dict, key=best_mapes_dict.get)})')
print(f' max MAPE: {max_mape:.2f}% ({max(best_mapes_dict, key=best_mapes_dict.get)})')
print()

# Airfare MAPES using 10-fold CV Parameter Grid Search on PreCOVID Data
# NOTE average airfare is from the lowest fare carrier

with open(r'./models/prophet_params_and_mapes/prophet_model_farelow_mapes_preCOVID.json', 'r') as file_in:
  best_mapes_dict = json.load(file_in)  # best_mapes is dict

mean_mape = 100 * mean(best_mapes_dict.values())
min_mape = 100 * min(best_mapes_dict.values())
max_mape = 100 * max(best_mapes_dict.values())

print(f'Best MAPES from Prophet CV Optimization for {len(best_mapes_dict)} Flights')
print(f'Note: This is for Lowest Airfare')
print(f' avg MAPE: {mean_mape:.2f}%')
print(f' min MAPE: {min_mape:.2f}% ({min(best_mapes_dict, key=best_mapes_dict.get)})')
print(f' max MAPE: {max_mape:.2f}% ({max(best_mapes_dict, key=best_mapes_dict.get)})')

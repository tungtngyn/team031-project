from bokeh.io import curdoc
from src.main import AirfarePredictionApp

layout = AirfarePredictionApp().build()

curdoc().add_root(layout)
curdoc().title = 'Team 31 Project'
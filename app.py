from bokeh.io import curdoc
from bokeh.settings import settings
settings.ico_path = r'./src/airfare.ico'


from src.main import AirfarePredictionApp

layout = AirfarePredictionApp().build()

curdoc().add_root(layout)
curdoc().title = 'Team 31 Project'
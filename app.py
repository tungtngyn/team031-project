from bokeh.io import curdoc
from bokeh.settings import settings
settings.ico_path = r'./src/favicon.ico'

from src.main import AirlineApp

layout = AirlineApp().build()

curdoc().add_root(layout)
curdoc().title = 'Team 31 Project'
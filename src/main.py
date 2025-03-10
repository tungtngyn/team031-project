from __future__ import annotations
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from bokeh.models.layouts import Row, Column, GridBox
    from bokeh.models.plots import GridPlot


import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.layouts import column, row, grid
from bokeh.models import Select, Button, ColumnDataSource, TableColumn, DataTable, Div, Spacer
from bokeh.models.css import Styles
from bokeh.sampledata.us_states import data as states


class AirlineApp():

    def __init__(self) -> None:

        # Default Margins
        self.default_margins = (20, 20, 20, 20)

        # Load data into memory
        self.df = pd.read_csv(r'./data/processed-data.csv')

        # Load CSS
        with open(r'./src/styles.css') as f:
            self.css = f.read()

        # Inputs
        self.dropdown_origin = Select(
            title='Origin', 
            value='',
            options=[''] + list(self.df['airport_name_concat_1'].unique()), 
            margin=self.default_margins, 
            width=200
        )
        self.dropdown_origin.on_change(
            'value', self._handle_origin_input_change
        )

        self.dropdown_destination = Select(
            title='Destination', 
            value='',
            options=[''] + list(self.df['airport_name_concat_2'].unique()), 
            margin=self.default_margins, 
            width=200
        )
        self.dropdown_destination.on_change(
            'value', self._handle_destination_input_change
        )

        self.dropdown_season = Select(
            title='Season of Travel', 
            value='', 
            options = [
                'Spring',
                'Summer',
                'Fall',
                'Winter'
            ], 
            margin=self.default_margins, 
            width=200
        )
        self.dropdown_season.on_change(
            'value', self._handle_season_input_change
        )

        self.dropdown_ml_model = Select(
            title='ML Model Selection', 
            value='', 
            margin=(0, 20, 0, 20), 
            options = [
                'Random Forest',
                'FB Prophet'
            ],
            width=200
        )
        self.dropdown_ml_model.on_change(
            'value', self._handle_ml_model_input_change
        )

        self.button_run_analyzer = Button(
            label='Analyze', 
            button_type='success', 
            margin=self.default_margins,
            width=200
        )

        self.button_update_charts = Button(
            label='Update Charts',
            button_type='success',
            margin=self.default_margins,
            width=200
        )

        self.analysis_results = Div(
            text="<h2>$   -  </h2>", 
            height=50, 
            width=100, 
            margin=(0, 0, 0, 20), 
            stylesheets=[self.css]
        )

        # Misc variables
        self.EXCLUDED = ('HI', 'AK') # Excluded states

        return None


    def _initialize_choropleth(self) -> None:
        """Function to initialize the choropleth chart. No data should be plotted yet.
        """

        state_xs = [states[code]["lons"] for code in states if code not in self.EXCLUDED]
        state_ys = [states[code]["lats"] for code in states if code not in self.EXCLUDED]

        self.choropleth = figure(
            title='Average Airfare Prices',
            height=400,
            width=800,
            x_axis_location=None, 
            y_axis_location=None,
            margin=self.default_margins
        )
        self.choropleth.grid.grid_line_color = None
        self.choropleth.patches(
            state_xs, 
            state_ys, 
            fill_alpha=0.0, 
            fill_color='#884444',
            line_color="#000000", 
            line_width=2, 
            line_alpha=0.3
        )

        return None


    def _initialize_histogram(self) -> None:
        """Function to initialize the histogram charts. No data should be plotted yet.
        """
        # Example: https://docs.bokeh.org/en/latest/docs/examples/topics/stats/histogram.html

        self.histogram = figure(
            title='Histogram',
            height=400,
            width=800,
            margin=self.default_margins
        )

        return None


    def _initialize_analyzer_charts(self) -> None:
        """Function to initialize the analysis charts. No data should be plotted yet.
        """
        # Example: https://docs.bokeh.org/en/latest/docs/examples/topics/stats/boxplot.html

        self.analyzer_charts = figure(
            title='Analyzer Charts',
            height=400,
            width=800,
            margin=self.default_margins
        )
        return None


    def _initialize_market_analysis_charts(self) -> None:
        """Function to initialize the market analysis charts. No data should be plotted yet.
        """
        self.market_analysis_charts = figure(
            title='Market Analysis Boxplot',
            height=400,
            width=550 - 15,
            margin=(0, 10, 0, 10)
        )

        self.market_analysis_table = DataTable(
            columns=[
                TableColumn(field='Airport'),
                TableColumn(field='Avg Price')
            ],
            height=400,
            width=250 - 15,
            margin=(0, 10, 0, 10)
        )
        return None


    def _update_choropleth(self) -> None:
        """Updates choropleth chart based on inputs
        """

        # Update circle at Origin

        # Update circle at Destination

        # Update state colors
        # state_colors = []
        # for code in states:
        #     if code not in self.EXCLUDED:
        #         state_colors.append()

        # self.choropleth.fill_color = state_colors

        return None


    def _update_histograms(self) -> None:
        return None


    def _update_market_analysis_charts(self) -> None:
        return None
    

    def _update_analyzer_charts(self) -> None:
        """Updates the analyzer charts when a new ML model is selected / run
        """
        return None
    

    def _update_analysis_results(self) -> None:
        self.analysis_results.text = '<h2>$ 130.00</h2>'
        return None
    

    def update_all_charts(self) -> None:
        """Update all charts when an input value is changed
        """
        self._update_choropleth()
        self._update_histograms()
        self._update_market_analysis_charts()

        return None
    

    def _handle_origin_input_change(self, attr, old, new) -> None:
        """Executed whenever the "Origin" airport changes
        """

        # Update Destination dropdown to relevant values
        if new == '':
            new_options = [''] + list(
                self.df['airport_name_concat_2'].unique()
            )
        else:
            new_options = [''] + list(
                self.df[self.df['airport_name_concat_1'] == new]['airport_name_concat_2'].unique()
            )

        self.dropdown_destination.options = new_options

        # Update charts
        self.update_all_charts()
        
        return None


    def _handle_destination_input_change(self, attr, old, new) -> None:
        """Executed whenever the "Destination" airport changes
        """

        # Update Origin dropdown to relevant values
        if new == '':
            new_options = [''] + list(
                self.df['airport_name_concat_1'].unique()
            )
        else:
            new_options = [''] + list(
                self.df[self.df['airport_name_concat_2'] == new]['airport_name_concat_1'].unique()
            )
        self.dropdown_origin.options = new_options

        # Update charts
        self.update_all_charts()

        return None


    def _handle_season_input_change(self, attr, old, new) -> None:
        """Executed whenever the "Season" changes
        """
        return None


    def _handle_ml_model_input_change(self, attr, old, new) -> None:
        """Executed whenever the "ML Model" selection changes
        """
        return None
    

    def build(self) -> Union[Row, Column, GridBox, GridPlot]:
        """Builds the webapp layout
        """

        # Build Components
        self._initialize_choropleth()
        self._initialize_histogram()
        self._initialize_analyzer_charts()
        self._initialize_market_analysis_charts()

        # Inputs & Controls
        controls = [
            self.dropdown_origin,
            self.dropdown_destination,
            self.dropdown_season
        ]
        inputs = row(
            *controls,
            sizing_mode='scale_width', 
            margin=(0, 20, 0, 20),
            styles=Styles(text_align='center', justify_content='center')
        )

        analyze_button = Button(label='Analyze', height=35, width=200, margin=(0, 200, 0, 20), align='end')
        analyze_button.on_click(self._update_analysis_results)
        analyzer_io = row(
            self.dropdown_ml_model, 
            analyze_button,
            self.analysis_results,
            margin=self.default_margins
        )

        # Divs, Titles, and Texts
        title = Div(
            text='<h1>Airfare Price Analyzer</h1>', 
            height=50,
            sizing_mode='stretch_width',
            styles=Styles(text_align='center', justify_content='center'),
            stylesheets=[self.css],
            margin=(0, 20, 20, 20)
        )

        table_title = Div(
            text='<p><b>Next 3 Closest Airports<b></p>', 
            height=30,
            sizing_mode='stretch_width',
            stylesheets=[self.css]
        )

        # Generate Layout
        layout = column(
            [
                title,
                inputs,
                # row(spacer_left, title, spacer_right, sizing_mode='scale_width'),
                # row(spacer_left, inputs, spacer_right, sizing_mode='scale_width'),
                row(self.choropleth, self.histogram, sizing_mode='scale_width'),
                row(
                    column(self.analyzer_charts, analyzer_io), 
                    row(
                        column(table_title, self.market_analysis_table), 
                        self.market_analysis_charts, 
                        margin=self.default_margins
                    ), sizing_mode='scale_width'
                )
            ],
            sizing_mode='scale_width'
        )

        return layout
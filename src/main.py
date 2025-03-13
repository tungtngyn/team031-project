from __future__ import annotations
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from bokeh.models.layouts import Row, Column, GridBox
    from bokeh.models.plots import GridPlot
    from bokeh.events import ButtonClick


import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import (
    Select, Button, ColumnDataSource, TableColumn, 
    DataTable, Div, ImageURL, LabelSet, Patches, 
    ColorBar, CategoricalColorMapper, HoverTool
)
from bokeh.models.css import Styles
from bokeh.sampledata.us_states import data as states
from bokeh.palettes import RdYlGn6 as palette


class AirfarePredictionApp():

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

        self.analysis_results = Div(
            text="<h2>$   -  </h2>", 
            height=50, 
            width=100, 
            margin=(0, 0, 0, 20), 
            stylesheets=[self.css]
        )

        # Misc variables
        self.EXCLUDED = ('HI', 'AK') # Excluded states

        # Calculate coordinates for all airports in dataset
        self.airport_coords = {
            row['airport_name_concat_1']: (row['longitude_1'], row['latitude_1']) 
            for _, row in 
                self.df[['airport_name_concat_1', 'longitude_1', 'latitude_1']]
                    .drop_duplicates()
                    .iterrows()
        }

        self.airport_coords.update({
            row['airport_name_concat_2']: (row['longitude_2'], row['latitude_2']) 
            for _, row in 
                self.df[['airport_name_concat_2', 'longitude_2', 'latitude_2']]
                    .drop_duplicates()
                    .iterrows()
        })

        # Calculate mapping between state abrvn & state name (ex. CA --> California)
        self.state_mapping = {k: v['name'] for k, v in states.items()}

        return None


    def _initialize_choropleth(self) -> None:
        """Function to initialize the choropleth chart. No data should be plotted yet.
        """

        # Figure for Choropleth
        self.choropleth = figure(
            title='Average Airfare Prices',
            height=400,
            width=800,
            x_axis_location=None, 
            y_axis_location=None,
            margin=self.default_margins
        )
        self.choropleth.grid.grid_line_color = None
        self.choropleth.toolbar.logo = None

        # State outlines
        state_xs = [states[code]["lons"] for code in states if code not in self.EXCLUDED]
        state_ys = [states[code]["lats"] for code in states if code not in self.EXCLUDED]
        self.choropleth_default_colors = ['#FFFFFF' for _ in range(len(state_xs))]
        self.choropleth_default_fares = ['-' for _ in range(len(state_xs))]
        
        self.choropleth_state_src = ColumnDataSource(dict(
            xs=state_xs,
            ys=state_ys,
            names=[self.state_mapping[code] for code in states if code not in self.EXCLUDED],
            abrvns=[code for code in states if code not in self.EXCLUDED],
            colors=self.choropleth_default_colors,
            avg_fares=self.choropleth_default_fares
        ))

        self.choropleth_patches = Patches(
            xs='xs', 
            ys='ys', 
            fill_color='colors',
            fill_alpha=0.7, 
            line_color='#000000', 
            line_alpha=0.3,
            line_width=2
        )
        self.choropleth.add_glyph(self.choropleth_state_src, self.choropleth_patches)        

        # Markers for origin / destination
        x0, y0 = self.airport_coords['SAN - San Diego International Airport']

        # Origin
        self.choropleth_origin_src = ColumnDataSource(dict(
            xs=[x0],
            ys=[y0],
            names=['Origin']
        ))

        self.choropleth_origin_marker = ImageURL(
            url={'value': 'https://cdn-icons-png.flaticon.com/128/18395/18395847.png'},
            x='xs',
            y='ys',
            w=2,
            h=2,
            global_alpha=0.0,
            anchor='center'
        )

        self.choropleth_origin_label = LabelSet(
            x='xs', 
            y='ys', 
            text='names',
            text_alpha=0.0,
            text_font_size='8pt',
            x_offset=-16,
            y_offset=-22, 
            source=self.choropleth_origin_src
        )

        self.choropleth.add_glyph(self.choropleth_origin_src, self.choropleth_origin_marker)
        self.choropleth.add_layout(self.choropleth_origin_label)

        # Destination
        self.choropleth_dest_src = ColumnDataSource(dict(
            xs=[x0],
            ys=[y0],
            names=['Destination']
        ))

        self.choropleth_dest_marker = ImageURL(
            url={'value': 'https://cdn-icons-png.flaticon.com/128/447/447031.png'},
            x='xs',
            y='ys',
            w=2,
            h=2,
            global_alpha=0.0,
            anchor='bottom'
        )

        self.choropleth_dest_label = LabelSet(
            x='xs', 
            y='ys', 
            text='names',
            text_alpha=0.0,
            text_font_size='8pt',
            x_offset=-28,
            y_offset=-20, 
            source=self.choropleth_dest_src
        )

        self.choropleth.add_glyph(self.choropleth_dest_src, self.choropleth_dest_marker)
        self.choropleth.add_layout(self.choropleth_dest_label)

        # Hover Tooltip
        hover = HoverTool(
            tooltips=[
                ("State", "@names"),
                ("Code", "@abrvns"),
                ("Avg. Fare", "@avg_fares")
            ], 
            mode='mouse'
        )
        self.choropleth.add_tools(hover)

        # Color Bar
        self.choropleth_default_mapper = CategoricalColorMapper(
            palette=['#D3D3D3'],
            factors=['$0.00']
        )

        self.choropleth_color_bar = ColorBar(
            color_mapper=self.choropleth_default_mapper,
            visible=True
        )
        
        self.choropleth.add_layout(self.choropleth_color_bar, 'below')

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
        self.histogram.toolbar.logo = None

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
        self.analyzer_charts.toolbar.logo = None

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
        self.market_analysis_charts.toolbar.logo = None

        self.market_analysis_table = DataTable(
            columns=[
                TableColumn(field='Origin Airport'),
                TableColumn(field='Avg. Price to Dest.')
            ],
            height=400,
            width=250 - 15,
            margin=(0, 10, 0, 10),
            index_position=None
        )
        return None


    def _update_choropleth(self) -> None:
        """Updates choropleth chart based on inputs
        """

        # Origin Value Updater
        origin_value = self.dropdown_origin.value
        if origin_value == '':
            self.choropleth_state_src.data['colors'] = self.choropleth_default_colors
            self.choropleth_state_src.data['avg_fares'] = self.choropleth_default_fares
            self.choropleth_origin_marker.global_alpha = 0.0
            self.choropleth_origin_label.text_alpha = 0.0
            self.choropleth_color_bar.update(color_mapper=self.choropleth_default_mapper)

        else:

            # Update origin marker
            x, y = self.airport_coords[origin_value]
            self.choropleth_origin_src.data['xs'] = [x]
            self.choropleth_origin_src.data['ys'] = [y]
            self.choropleth_origin_marker.global_alpha = 1.0
            self.choropleth_origin_label.text_alpha = 1.0

            # Filter / aggregate data
            df = (
                self.df[(self.df['airport_name_concat_1'] == origin_value)]
                    .groupby('state_2')
                    .agg({'fare': 'mean'})
            )

            fares = df.to_dict(orient='index')
            
            # Update state colors
            bins = np.linspace(
                df['fare'].min() - 0.001, # account for float roundoff error
                df['fare'].max() + 0.001, 
                num=7
            )

            d = pd.cut(
                df['fare'], 
                bins=bins, 
                labels=list(range(6))
            ).to_dict()

            if df.shape[0] > 0:

                # Calculate colors
                state_colors = []
                avg_fares = []

                for code in states:
                    if code not in self.EXCLUDED:
                        name = self.state_mapping[code]
                        
                        if name in d:
                            state_colors.append(palette[d[name]])
                            avg_fares.append(f"${fares[name]['fare']:.2f}")

                        else:
                            state_colors.append('#FFFFFF')
                            avg_fares.append('-')

                self.choropleth_state_src.data['colors'] = state_colors
                self.choropleth_state_src.data['avg_fares'] = avg_fares

                # Update colorbar
                new_color_mapper = CategoricalColorMapper(
                    palette=palette,
                    factors=[
                        f'${0.5*(bins[i] + bins[i+1]):.2f}' for i in range(6)
                    ]
                )
                self.choropleth_color_bar.update(color_mapper=new_color_mapper)

            else:

                # Hide / Reset if no data available
                self.choropleth_state_src.data['colors'] = self.choropleth_default_colors
                self.choropleth_state_src.data['avg_fares'] = self.choropleth_default_fares
                self.choropleth_color_bar.udpate(color_mapper=self.choropleth_default_mapper)


        # Destination Value Updater
        destination_value = self.dropdown_destination.value
        if destination_value == '':
            self.choropleth_dest_marker.global_alpha = 0.0
            self.choropleth_dest_label.text_alpha = 0.0

        else:
            # Update destination marker
            x, y = self.airport_coords[destination_value]
            self.choropleth_dest_src.data['xs'] = [x]
            self.choropleth_dest_src.data['ys'] = [y]
            self.choropleth_dest_marker.global_alpha = 1.0
            self.choropleth_dest_label.text_alpha = 1.0

        return None


    def _update_histograms(self) -> None:
        """Updates the histogram charts when a new origin/destination is selected.
        """
        return None


    def _update_market_analysis_charts(self) -> None:
        """Updates the market analysis charts when a new origin/destination is selected.
        """
        return None
    

    def _update_analyzer_charts(self) -> None:
        """Updates the analyzer charts when a new ML model is selected / run
        """
        return None
    

    def _update_analysis_results(self, value: float) -> None:
        self.analysis_results.text = f'<h2>$ {value:.2f}</h2>'
        return None
    

    def _handle_origin_input_change(self, attr: str, old: str, new: str) -> None:
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
        self._update_choropleth()
        self._update_histograms()
        self._update_market_analysis_charts()

        return None


    def _handle_destination_input_change(self, attr: str, old: str, new: str) -> None:
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
        self._update_choropleth()
        self._update_histograms()
        self._update_market_analysis_charts()

        return None


    def _handle_season_input_change(self, attr: str, old: str, new: str) -> None:
        """Executed whenever the "Season" changes
        """

        # Reset prediction
        self.analysis_results.text="<h2>$   -  </h2>"

        return None


    def _handle_ml_model_input_change(self, attr: str, old: str, new: str) -> None:
        """Executed whenever the "ML Model" selection changes
        """

        # Reset prediction
        self.analysis_results.text="<h2>$   -  </h2>"
        
        return None


    def _handle_analyze_button_click(self, event: ButtonClick) -> None:
        """Executed whenever the "Analyze" button is clicked
        """

        # Get data for inference. Full processed data available in self.df
        origin = self.dropdown_origin.value           # returns "airport_name_concat_1" (SAN - San Diego International Airport)
        destination = self.dropdown_destination.value # returns "airport_name_concat_2"
        season = self.dropdown_season.value           # returns "season"

        # Perform inference
        # ...
        estimated_price = 130.00 # update this

        # Update analysis charts
        # ...

        # Update analysis results
        self._update_analysis_results(estimated_price)

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
        analyze_button.on_click(self._handle_analyze_button_click)
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
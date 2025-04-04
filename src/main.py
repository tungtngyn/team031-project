from __future__ import annotations
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from bokeh.models.layouts import Row, Column, GridBox
    from bokeh.models.plots import GridPlot
    from bokeh.events import ButtonClick


import pandas as pd
import numpy as np
from functools import partial
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import (
    Select, Button, ColumnDataSource, Div, ImageURL, LabelSet, Patches, 
    ColorBar, CategoricalColorMapper, HoverTool, Range1d, FactorRange,
    TabPanel, Tabs
)
from bokeh.models.css import Styles
from bokeh.sampledata.us_states import data as states
from bokeh.palettes import RdYlGn6 as palette

from prophet import Prophet
from dev.fbp_tsa import (
    find_longest_timeseq, fbp_predict_future
) # import custom user functions
import matplotlib.pyplot as plt

import holoviews as hv
from holoviews.streams import Stream

hv.extension('bokeh')


# Custom streamer to update HoloViews DynamicMap plots
class UpdateStream(Stream):
    def event(self, **kwargs):
        super().event(**kwargs) # Triggers the update


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
                'All',
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
                '',
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
        self.multi_airport_codes = {'ACY', 'ORD', 'DTW', 'LGA', 'IAD', 'EGE'} # IATA codes with multiple airports after remapping
        self.ts_cols = ['year', 'quarter', 'airport_iata_1', 'airport_iata_2', 'fare'] # time series columns

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

        # Bar chart ylim
        self.bar_chart_xlim = {}
        self.bar_chart_ylim = {}

        # Analysis Model
        self.prophet_df = None  # store Prophet pd.DataFrame

        return None


    # ----------------------------------------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------------------------------------
    def _update_analysis_results(self, value: float) -> None:
        self.analysis_results.text = f'<h2>$ {value:.2f}</h2>'
        return None


    def _get_filtered_data(self) -> pd.DataFrame:
        """Returns a filtered dataframe based on the options selected
        """

        filtered_df = self.df.copy(deep=True)

        if self.dropdown_origin.value != '':
            filtered_df = filtered_df[
                filtered_df['airport_name_concat_1'] == self.dropdown_origin.value
            ]

        if self.dropdown_destination.value != '':
            filtered_df = filtered_df[
                filtered_df['airport_name_concat_2'] == self.dropdown_destination.value
            ]

        if self.dropdown_season.value != '' and self.dropdown_season != 'All':
            filtered_df = filtered_df[
                filtered_df['season'] == self.dropdown_season.value
            ]

        return filtered_df


    # ----------------------------------------------------------------------------------------------
    # Choropleth Functions
    # ----------------------------------------------------------------------------------------------
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
            y_offset=-20, 
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
            y_offset=-14, 
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


    # ----------------------------------------------------------------------------------------------
    # Histogram Functions
    # ----------------------------------------------------------------------------------------------
    def _redraw_holoviews_histogram(self) -> hv.Histogram:
        """Function to redefine the histogram using the HoloViews API
        """

        # Tooltip
        hover_tooltips = [
            ('Frequency', '@top')
        ]
        
        # Initialize with invisible chart
        if any([self.dropdown_origin.value == '', self.dropdown_destination == '']):
            alpha = 0.
            
        else:
            alpha = 0.8

        # Get filtered data
        filtered_df = self._get_filtered_data()

        # Calculate histogram
        bins = np.arange(filtered_df["fare"].min(), filtered_df["fare"].max() + 20, 20)
        hist, edges = np.histogram(filtered_df["fare"], bins=bins)

        # Update limits based on values
        self.histogram_xlim = Range1d(edges[0] - 20, edges[-1] + 20)
        self.histogram_ylim = Range1d(0, np.max(hist) + 20)

        return hv.Histogram((edges, hist)).opts(
            xlabel="Fare (Large Market Share)",
            ylabel="Frequency",
            title=f"Fare Distribution",
            width=800, 
            height=400, 
            tools=["hover"],
            color="#aec7e8", 
            line_color="black", 
            alpha=alpha,
            hover_tooltips=hover_tooltips,
            margin=self.default_margins
        )


    def _initialize_histogram(self) -> None:
        """Function to initialize the histogram charts.
        """
        self.hv_histogram = hv.DynamicMap(self._redraw_holoviews_histogram, streams=[UpdateStream()])
        self.bk_histogram = hv.render(self.hv_histogram)
        self.bk_histogram.toolbar.logo = None
        return None


    def _update_histogram(self) -> None:
        """Updates the histogram chart when new options are selected.
        """
        self.hv_histogram.event() # Trigger a redraw, recalculate xlim, ylim
        self.bk_histogram.x_range = self.histogram_xlim
        self.bk_histogram.y_range = self.histogram_ylim
        return None
    

    # ----------------------------------------------------------------------------------------------
    # Line Chart Functions
    # ----------------------------------------------------------------------------------------------
    def _redraw_holoviews_line_chart(self) -> hv.Overlay:
        """Function to redefine the line chart using the HoloViews API
        """

        # Tooltips
        hover_tooltips = [
            ('Year', '@year'),
            ('Avg. Fare', '$@{avg_fare}{0.2f}'),
            ('Avg. Fare (Low)', '$@{avg_fare_low}{0.2f}'),
            ('Avg. Fare (Lg)', '$@{avg_fare_lg}{0.2f}')
        ]
        
        # Initialize with invisible chart
        if any([self.dropdown_origin.value == '', self.dropdown_destination == '']):
            alpha = 0.
            
        else:
            alpha = 0.8

        # Get filtered data
        filtered_df = self._get_filtered_data()

        # Aggregate
        avg_fare_by_year = filtered_df.groupby("year").agg(
            avg_fare=("fare", "mean"),
            avg_fare_lg=("fare_lg", "mean"),
            avg_fare_low=("fare_low", "mean")
        ).reset_index()
        
        # Update limits based on values
        self.line_chart_xlim = Range1d(filtered_df['year'].min(), filtered_df['year'].max())
        self.line_chart_ylim = Range1d(0, avg_fare_by_year['avg_fare'].max() + 20)

        # Line charts
        avg_fare_line = hv.Curve(
            avg_fare_by_year, 
            "year", 
            "avg_fare", 
            label="Avg Fare"
        ).opts(
            line_color="blue", 
            line_width=2, 
            tools=["hover"], 
            alpha=alpha, 
            hover_tooltips=hover_tooltips
        )

        avg_fare_lg_line = hv.Curve(
            avg_fare_by_year, 
            "year", 
            "avg_fare_lg", 
            label="Avg Fare (Largest Airline)"
        ).opts(
            line_color="#D3D3D3", 
            line_width=2, 
            line_dash="dashed", 
            tools=["hover"], 
            alpha=alpha, 
            hover_tooltips=hover_tooltips
        )

        avg_fare_low_line = hv.Curve(
            avg_fare_by_year, 
            "year", 
            "avg_fare_low", 
            label="Avg Fare (Lowest-Cost Airline)"
        ).opts(
            line_color="#808080", 
            line_width=2, 
            line_dash="dotted", 
            tools=["hover"],
            alpha=alpha, 
            hover_tooltips=hover_tooltips
        )

        return (avg_fare_line * avg_fare_lg_line * avg_fare_low_line).opts(
            xlabel="Year", 
            ylabel="Average Fare",
            title=f"Average Fares by Year",
            width=800, 
            height=400, 
            legend_position="bottom_left", 
            tools=["hover"],
            margin=self.default_margins
        )
    

    def _initialize_line_chart(self) -> None:
        """Function to initialize the line charts.
        """
        self.hv_line_chart = hv.DynamicMap(self._redraw_holoviews_line_chart, streams=[UpdateStream()])
        self.bk_line_chart = hv.render(self.hv_line_chart)
        self.bk_line_chart.toolbar.logo = None
        return None


    def _update_line_chart(self) -> None:
        """Updates the line chart when new options are selected.
        """
        self.hv_line_chart.event() # Trigger a redraw, recalculate xlim, ylim
        self.bk_line_chart.x_range = self.line_chart_xlim
        self.bk_line_chart.y_range = self.line_chart_ylim
        return None
    

    # ----------------------------------------------------------------------------------------------
    # Seasonal Boxplot Functions
    # ----------------------------------------------------------------------------------------------
    def _redraw_holoviews_seasonal_boxplot(self) -> hv.Overlay:
        """Function to redefine the seasonal boxplot using the HoloViews API
        """
        
        # Initialize with invisible chart
        if any([self.dropdown_origin.value == '', self.dropdown_destination == '']):
            alpha = 0.
            
        else:
            alpha = 0.8

        # Get filtered data
        filtered_df = self._get_filtered_data()

        # Define the correct season order
        season_order = ['Spring', 'Summer', 'Fall', 'Winter']
        
        # Convert season to categorical with specified order
        filtered_df['season'] = pd.Categorical(
            filtered_df['season'],
            categories=season_order,
            ordered=True
        )
        
        # Sort by season
        filtered_df = filtered_df.sort_values('season')
        
        # Update limits based on values
        self.seasonal_boxplot_ylim = Range1d(0, filtered_df['fare'].max() + 20)

        return hv.BoxWhisker(
            filtered_df,
            'season',
            'fare'
        ).opts(
            title=f"Seasonal Fare Distribution",
            width=800, 
            height=400,
            ylabel="Fare ($)", 
            xlabel="Season",
            box_color='season',
            cmap='Category10',
            show_legend=False,
            box_alpha=alpha,
            box_line_alpha=alpha,
            outlier_alpha=alpha,
            whisker_alpha=alpha,
            margin=self.default_margins
        )
    

    def _initialize_seasonal_boxplot(self) -> None:
        """Function to initialize the seasonal boxplot.
        """
        self.hv_seasonal_boxplot = hv.DynamicMap(self._redraw_holoviews_seasonal_boxplot, streams=[UpdateStream()])
        self.bk_seasonal_boxplot = hv.render(self.hv_seasonal_boxplot)
        self.bk_seasonal_boxplot.toolbar.logo = None
        return None


    def _update_seasonal_boxplot(self) -> None:
        """Updates the seasonal boxplot when new options are selected.
        """
        self.hv_seasonal_boxplot.event() # Trigger a redraw, recalculate ylim
        self.bk_seasonal_boxplot.y_range = self.seasonal_boxplot_ylim
        return None
    

    # ----------------------------------------------------------------------------------------------
    # Airline Chart Functions
    # ----------------------------------------------------------------------------------------------
    def _redraw_holoviews_bar_chart(self, carrier_type: str) -> hv.Bars:
        """Function to redefine the bar chart using the HoloViews API
        """
        
        # Tooltips
        hover_tooltips = [
            ('Avg. Fare', '$@{avg_fare}{0.2f}')
        ]

        # Initialize with invisible chart
        if any([self.dropdown_origin.value == '', self.dropdown_destination == '']):
            alpha = 0.
            
        else:
            alpha = 0.8

        # Get filtered data
        filtered_df = self._get_filtered_data()

        # Aggregate
        col = 'fare_lg' if carrier_type == 'lg' else 'fare_low'
        name_col = 'carrier_lg_name_concat' if carrier_type == 'lg' else 'carrier_low_name_concat'
        
        avg_data = filtered_df.groupby(name_col).agg(avg_fare=(col, "mean")).reset_index()
        avg_data = avg_data.sort_values("avg_fare", ascending=False)[:10].reset_index(drop=True)

        # Update limits based on values
        self.bar_chart_xlim[carrier_type] = Range1d(0, avg_data['avg_fare'].max() + 20)
        self.bar_chart_ylim[carrier_type] = FactorRange(*avg_data[name_col].to_list())
    
        return hv.Bars(avg_data, name_col, "avg_fare").opts(
            xlabel=f"Average Fare ({'Largest' if carrier_type=='lg' else 'Lowest-Cost'} Carrier)",
            ylabel="Airline",
            title=f"Average Fare ({'Largest' if carrier_type=='lg' else 'Lowest-Cost'} Carrier) by Airline",
            width=800, 
            height=400, 
            tools=["hover"],
            invert_axes=True,
            color="#1f77b4" if carrier_type == 'lg' else "#ff7f0e",
            alpha=alpha,
            hover_tooltips=hover_tooltips,
            margin=self.default_margins
        )


    def _initialize_lg_bar_chart(self) -> None:
        """Function to initialize the "lg" carrier bar chart.
        """
        self.hv_lg_bar_chart = hv.DynamicMap(partial(self._redraw_holoviews_bar_chart, carrier_type='lg'), streams=[UpdateStream()])
        self.bk_lg_bar_chart = hv.render(self.hv_lg_bar_chart)
        self.bk_lg_bar_chart.toolbar.logo = None
        return None


    def _update_lg_bar_chart(self) -> None:
        """Updates the "lg" carrier bar chart when new options are selected.
        """
        self.hv_lg_bar_chart.event() # Trigger a redraw, recalculate ylim
        self.bk_lg_bar_chart.x_range = self.bar_chart_xlim['lg']
        self.bk_lg_bar_chart.y_range = self.bar_chart_ylim['lg']
        return None


    def _initialize_low_bar_chart(self) -> None:
        """Function to initialize the "low" carrier bar chart.
        """
        self.hv_low_bar_chart = hv.DynamicMap(partial(self._redraw_holoviews_bar_chart, carrier_type='low'), streams=[UpdateStream()])
        self.bk_low_bar_chart = hv.render(self.hv_low_bar_chart)
        self.bk_low_bar_chart.toolbar.logo = None
        return None


    def _update_low_bar_chart(self) -> None:
        """Updates the "low" carrier bar chart when new options are selected.
        """
        self.hv_low_bar_chart.event() # Trigger a redraw, recalculate ylim
        self.bk_low_bar_chart.x_range = self.bar_chart_xlim['low']
        self.bk_low_bar_chart.y_range = self.bar_chart_ylim['low']
        return None
    

    # ----------------------------------------------------------------------------------------------
    # Input Change Callback Functions
    # ----------------------------------------------------------------------------------------------
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
        self._update_histogram()
        self._update_line_chart()
        self._update_seasonal_boxplot()
        self._update_lg_bar_chart()
        self._update_low_bar_chart()

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
        self._update_histogram()
        self._update_line_chart()
        self._update_seasonal_boxplot()
        self._update_lg_bar_chart()
        self._update_low_bar_chart()

        return None


    def _handle_season_input_change(self, attr: str, old: str, new: str) -> None:
        """Executed whenever the "Season" changes
        """

        # Reset prediction
        self.analysis_results.text="<h2>$   -  </h2>"

        # Update charts
        self._update_histogram()
        self._update_line_chart()
        self._update_lg_bar_chart()
        self._update_low_bar_chart()

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

        # FB Prophet Time Series Forecasting
        if self.dropdown_ml_model.value == 'FB Prophet':
            # Check for Valid Inputs (note Prophet analysis needs origin and destination input)
            if (origin == '') or (destination == ''):
                print('Please select valid Origin and Destination from dropdown.')
            else:
                src, dst = origin.split()[0], destination.split()[0] # get 3 digit code
                ts_df = self._get_filtered_data()[self.ts_cols] # filter to route data and desired cols
                # Check if Data has 'year' ending in 2024
                if ts_df['year'].max() != 2024:
                    print('Selected route ineligible for FB Prophet forecasting.')
                    print('Please select a different route.')
                    return None
                ts_df['date'] = ts_df['year'].astype(str) \
                    .str.cat(ts_df['quarter'].astype(str), sep='-Q')  # example output '2024-Q1'
                # Check if Data needs to be Aggregated for Codes that have Multiple Airports
                if {src, dst}.intersection(self.multi_airport_codes): # if non-empty set intersection
                    ts_df = ts_df.groupby(['date'], as_index=False, sort=False).agg({'fare': 'mean'})
                ts_df = ts_df.sort_values(by='date').reset_index(drop=True)
                # Extract the Longest Nonbreaking Sequence of Quarterly Dates
                ts_df = find_longest_timeseq(data=ts_df, ycol='fare', min_rows=50) # returns DF or None
                # Check if Data has at least 50 rows for Forecasting
                if ts_df is None:
                    print('Selected route ineligible for FB Prophet forecasting.')
                    print('Please select a different route.')
                    return None
                # Use Prophet to make Predictions 8 Qtrs Ahead from 2024-Q1 to 2026-Q1
                m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=2)
                fcst_df = fbp_predict_future(model=m, data=ts_df, n=8)
                self.prophet_df = pd.concat([ts_df,
                                            fcst_df[['ds', 'yhat']].rename(columns={'yhat': 'y'})],
                                            ignore_index=True) # append fcst_df to ts_df
                # Example Matplotlib Plot
                # self.prophet_df.plot.line(x='ds', y='y', color='blue',
                #                             title='FB Prophet Forecasting',
                #                             xlabel='Date', ylabel='Fare',
                #                             linestyle='-', linewidth=1,
                #                             marker='o',
                #                             label=f'{src}-{dst}')
                # print(self.prophet_df)
                # plt.show()

        # Update analysis charts
        # ...

        # Update analysis results
        self._update_analysis_results(estimated_price)

        return None
    

    # ----------------------------------------------------------------------------------------------
    # Webapp Layout
    # ----------------------------------------------------------------------------------------------
    def build(self) -> Union[Row, Column, GridBox, GridPlot]:
        """Builds the webapp layout
        """

        # Build Components
        self._initialize_choropleth()
        self._initialize_histogram()
        self._initialize_line_chart()
        self._initialize_seasonal_boxplot()
        self._initialize_lg_bar_chart()
        self._initialize_low_bar_chart()

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

        # Generate Layout
        t1 = column(self.bk_histogram, self.bk_seasonal_boxplot)
        t2 = column(self.bk_lg_bar_chart, self.bk_low_bar_chart)

        tabs = Tabs(
            tabs=[
                TabPanel(child=t1, title="Analysis by Fare"),
                TabPanel(child=t2, title="Analysis by Airline")
            ]
        )

        layout = column(
            [
                title,
                inputs,
                row(
                    column(self.choropleth, self.bk_line_chart, analyzer_io), 
                    tabs, 
                    sizing_mode='scale_width'
                )
            ],
            sizing_mode='scale_width'
        )

        return layout
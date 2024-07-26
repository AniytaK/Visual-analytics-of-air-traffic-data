import pandas as pd
from dash import Dash, dcc, html, State, callback_context, dash_table
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import STL
import re


# Data import
from Flight_Data_preprocessing import (combine_airport_traffic_data, combine_airport_routes_data,
                                       airport_location_coordinates,top_5_airports_data_for_line_chart,
                                       combine_flight_efficiency_data, combine_airlines_routes_data)

traffic_europe = combine_airport_traffic_data()
airports_routes = combine_airport_routes_data()
airports_location = airport_location_coordinates()
top_5_data = top_5_airports_data_for_line_chart(traffic_europe, 2023, country=None)
horizontal_flight_eff = combine_flight_efficiency_data()
airlines_routes_data = combine_airlines_routes_data()

#Funktions import
from Functions_for_Dashboard import (top_5_airports_by_year, create_map_with_airports,
                                     create_map_with_airports_2, create_map_with_airlines, flight_traffic_data_for_country,
                                    calculate_mse, get_most_similar_airports, generate_heatmap_data,
                                     get_flight_info, data_pie_chart)
def normalize_series(series):
    return (series - series.min()) / (series.max() - series.min()) * 100

traffic_europe['FLT_TOT_1_norm'] = traffic_europe.groupby('APT_NAME')['FLT_TOT_1'].transform(normalize_series)
traffic_europe = traffic_europe.dropna(subset=['FLT_TOT_1_norm'])
def filter_alphabets_only(word):
    return re.match("^[A-Za-z\s]+$", word) is not None


########################################################################################################################
app = Dash(__name__, external_stylesheets=[dbc.themes.SIMPLEX])

app.layout = html.Div([
    dbc.Container(fluid=True, children=[
        dbc.Row([
            dbc.Col(html.H1('Flight Analysis',
                            style={'font-family': 'Open Sans', 'font-size': '38px', 'font-weight': 'bold',
                                   'color': '#FFFFFF'},
                            className='mt-1 d-flex align-items-end'), width=9),
        ], className='mt-1 d-flex align-items-end',
            style={'background-color': '#2A52BE', 'height': '55px', 'border-radius': '0px'}
        ),
        html.Div(style={'height': '3px'}),
        # _____________________________________________________________________________________________________________
        dbc.Row([
            dbc.Col(html.Div([
                html.B('Filters',
                       style={'font-family': 'Open Sans', 'font-size': '24px', 'font-weight': 'bold',
                              'color': '#FFFFFF', 'margin-right': '5px', 'margin-left': '7px'}),
                dcc.Dropdown(
                    id='year-dropdown',
                    options=[{'label': str(year), 'value': year} for year in range(2016, 2025)],
                    value=2023,  # Default year
                    style={'height': '4vh', 'margin-left': '3px', 'margin-right': '7px', 'font-family': 'Open Sans',}
                ),
                html.Div(style={'height': '10px'}),
                dcc.Dropdown(
                    id='country-dropdown',
                    options=[{'label': country, 'value': country} for country in
                             sorted(traffic_europe["STATE_NAME"].unique().tolist())],
                    placeholder="Select a country",
                    style={'height': '4vh', 'margin-left': '3px', 'margin-right': '7px', 'font-family': 'Open Sans',}
                ),
                html.Div(style={'height': '10px'}),
                dcc.Dropdown(
                    id='airport-dropdown',
                    options=[],
                    placeholder="Select an airport",
                    style={'height': '4vh', 'margin-left': '3px', 'margin-right': '7px', 'font-family': 'Open Sans',}
                ),
                html.Div(style={'height': '10px'}),
                dcc.Dropdown(
                    id='airline-dropdown',
                    options=[{'label': airline, 'value': airline} for airline in sorted(airlines_routes_data['Airline'].unique()) if filter_alphabets_only(airline)],
                    placeholder="Select an airline",
                    style={'height': '4vh', 'margin-left': '3px', 'margin-right': '7px', 'font-family': 'Open Sans',}
                ),
                html.Div(style={'height': '10px'}),

                html.Button('Reset', id='reset-button', n_clicks=0,
                            style={'height': '4vh', "width": "100px", 'background-color': '#0000FF', 'color': '#FFFFFF',
                                   'margin-left': '10px', 'margin-right': '3px', 'font-family': 'Open Sans', 'font-weight': 'bold'}),
                html.Div(style={'height': '40px'}),

                html.Div(id='airport-info',
                         style={'background-color': '59697a', 'color': '#FFFFFF',
                                'padding': '10px', 'border-radius': '10px', 'display': 'none', 'font-family': 'Open Sans',}),
                html.Div(style={'height': '20px'}),

                html.Div(id='realtime_airport-info',
                         style={'background-color': '#73ACE4', 'color': '#FFFFFF',
                                'padding': '10px', 'border-radius': '10px', 'display': 'none', 'font-family': 'Open Sans', }),
            ], style={'height': '840px', "background-color": "#002B36", 'border-radius': '10px', }),
                xs=12, sm=12, md=4, lg=3, xl=2
            ),
            # _________________________________________________________________________________________________________
            dbc.Col([
                dbc.Tabs(className='custom-tabs', children=[
                    dbc.Tab(label='Airport Analytics', tabClassName='custom-tab', activeTabClassName='custom-tab--selected', children=[
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id='bar-chart', style={'width': '100%', 'height': '100%', 'font-family': 'Open Sans',})
                            ], xs=12, sm=12, md=6, lg=4, xl=4, style={'background-color': "#E3E9EF"}
                            ),
                            dbc.Col([
                                dbc.Row([
                                    html.Div(
                                        html.Iframe(id='map',
                                                    style={
                                                        'height': '500px',
                                                        'width': '180%',
                                                        'border': 'none',
                                                        'margin-left': '0px',
                                                    }
                                        ),
                                        style={'height': '100%', 'width': '100%', 'margin-left': '0px'}
                                    ),
                                ], style={'height': '400px', 'width': '100%',}),
                            ], xs=12, sm=12, md=6, lg=4, xl=4, style={'background-color': "#E3E9EF", 'height': '100%'}),

                            dbc.Col([
                                html.Div(id='heatmap', style={'width': '100%', 'height': '100%', 'font-family': 'Open Sans',})
                            ], xs=12, sm=12, md=12, lg=4, xl=4, style={'background-color': "#E3E9EF"}
                            ),
                        ], style={'height': '400px',
                                  'margin-top': '5px',
                                  'margin-bottom': '0px',
                                  'border-radius': '0px',
                                  'background-color': '#E3E9EF',
                                  'font-family': 'Open Sans'}),
                        html.Div(style={'height': '6px'}),
                        dbc.Row([
                            dbc.Col(html.Button('Switch to Heatmap', id='restyle-button-heatmap', n_clicks=0,
                                                style={'display': 'inline-block', 'vertical-align': 'middle',
                                                       "background-color": "#eff3ff", 'font-family': 'Open Sans',
                                                       'border-radius': '100px',
                                                       "width": "100%",
                                                       'height': "25px"}
                                                ),
                                    width=6),
                            dbc.Col(html.Button('Switch to Linechart', id='restyle-button-linechart', n_clicks=0,
                                                style={'display': 'inline-block', 'vertical-align': 'middle',
                                                       "background-color": "#eff3ff", 'font-family': 'Open Sans',
                                                       'border-radius': '100px',
                                                       "width": "100%",
                                                       'height': "25px"}),
                                    width=6),
                        ]),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id='line-chart', style={'width': '100%', 'height': '100%',
                                                                      'font-family': 'Open Sans',}), width=12)
                        ], style={
                            'height': '340px',
                            'margin-top': '0px',
                            'border-radius': '0px',
                            'background-color': '#E3E9EF',
                            'font-family': 'Open Sans',
                            'font-size': '12px'
                        }),
                    ]),
                    # ___________________________________________________________________________________________________
                    dbc.Tab(label='Airline Analytics', tabClassName='custom-tab',
                            activeTabClassName='custom-tab--selected', children=[
                        dbc.Col([
                            dbc.Row([
                                html.Iframe(id='map1',
                                            style={
                                                'height': '470px',
                                                'background-color': '#E3E9EF',
                                                'border-radius': '0px',
                                                'margin': '0px',
                                            }
                                            ),
                            ]),
                            html.Div(style={'height': '8px'}),
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id='bar-chart-largest-airports', style={'width': '100%', 'height': '100%', 'font-family': 'Open Sans',})
                                ], xs=12, sm=12, md=6, lg=4, xl=4,
                                    style={'height': '280px', 'margin-top': '0px',
                                           'border-radius': '0px',
                                           'background-color': '#E3E9EF',
                                           'font-family': 'Open Sans', 'font-size': '12px'}
                                ),
                                dbc.Col([
                                    dcc.Graph(id='treemap', style={'width': '100%', 'height': '100%', 'font-family': 'Open Sans',})
                                ], xs=12, sm=12, md=6, lg=4, xl=4,
                                    style={'height': '280px', 'margin-top': '0px',
                                           'border-radius': '0px',
                                           'background-color': '#E3E9EF',
                                           'font-family': 'Open Sans', 'font-size': '12px'}
                                ),
                                dbc.Col([
                                    dcc.Graph(id='pie', style={'width': '100%', 'height': '100%', 'font-family': 'Open Sans',})
                                ], xs=12, sm=12, md=6, lg=4, xl=4,
                                    style={'height': '280px', 'margin-top': '0px',
                                           'border-radius': '0px',
                                           'background-color': '#E3E9EF',
                                           'font-family': 'Open Sans', 'font-size': '12px'}
                                ),
                            ])
                        ], width=12, style={'background-color': "#E3E9EF"}),
                    ])
                ])
            ], xs=12, sm=12, md=8, lg=9, xl=10)
        ], style={'background-color': "#E3E9EF"})
    ])
])
########################################################################################################################

#Tab 1
@app.callback(
    Output('bar-chart', 'figure'),
    [Input('year-dropdown', 'value'),
     Input('country-dropdown', 'value'),
     Input('airport-dropdown', 'value')]
)
def update_bar_chart(selected_year, selected_country, selected_airport):
    if selected_airport:
        airports_df = get_most_similar_airports(traffic_europe, selected_airport, column='FLT_TOT_1_norm')
        if airports_df.empty:
            raise PreventUpdate  # Prevent updating if the dataframe is empty

        fig = px.bar(
            airports_df,
            y=airports_df['MSE_TOT'],
            x=airports_df['Airport'],
            title=f'Most similar airports to {selected_airport}<br>in terms of its flight frequency',
            labels={'MSE_TOT': 'MSE', 'Airport': 'Airport'},
            color='Airport',
            color_discrete_sequence=px.colors.sequential.YlGnBu_r,
            template='plotly_white'
        )
        fig.update_layout(
            showlegend=False,
            height=400,
            width=400,
            title={
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 16},
            },
            margin={'t': 60}
        )
    elif selected_country:
        top_5_airports = top_5_airports_by_year(traffic_europe, selected_year, selected_country)

        fig = px.bar(
            top_5_airports,
            x='APT_NAME',
            y='FLT_TOT_1',
            title=f"Top Airports of {selected_country}",
            labels={'APT_NAME': 'Airport', 'FLT_TOT_1': 'Total flights'},
            color='APT_NAME',
            color_discrete_sequence=px.colors.sequential.YlGnBu_r,
            template='plotly_white'
        )

        fig.update_layout(
            showlegend=False,
            height=400,
            width=400,
            title={
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 16}
            },
            margin={'t': 60}
        )
    else:
        top_5_airports=top_5_airports_by_year(traffic_europe, selected_year, selected_country)

        fig = px.bar(
            top_5_airports,
            x='APT_NAME',
            y='FLT_TOT_1',
            title=f"Top Airports of {selected_year}",
            labels={'APT_NAME': 'Airport', 'FLT_TOT_1': 'Total flights'},
            color='APT_NAME',
            color_discrete_sequence=px.colors.sequential.YlGnBu_r,
            template='plotly_white'
        )
        fig.update_layout(
            showlegend=False,
            height=400,
            width=400,
            title={
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 16}
            },
            margin={'t': 60}
        )
    return fig

@app.callback(
    Output('map', 'srcDoc'),
    [Input('country-dropdown', 'value'),
     Input('airport-dropdown', 'value')]
)
def update_map(selected_country, selected_airport):
    map_object = create_map_with_airports(airports_location, airports_routes, selected_country, selected_airport)
    return map_object._repr_html_()

@app.callback(
    Output('line-chart', 'figure'),
    [Input('year-dropdown', 'value'),
     Input('country-dropdown', 'value'),
     Input('airport-dropdown', 'value'),
     Input('restyle-button-heatmap', 'n_clicks'),
     Input('restyle-button-linechart', 'n_clicks')],
    [State('line-chart', 'figure')]
)
def update_line_chart(selected_year, selected_country, selected_airport, n_clicks_heatmap, n_clicks_linechart, current_fig):
    # Determine which button was clicked
    ctx = callback_context
    if not ctx.triggered:
        button_id = 'restyle-button-linechart'  # Default to line chart
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if selected_airport:
        filtered_traffic_data = traffic_europe.loc[
            (traffic_europe['APT_NAME'] == selected_airport) & (traffic_europe['YEAR'] == selected_year)]

        filtered_traffic_data['FLT_DATE'] = pd.to_datetime(filtered_traffic_data['FLT_DATE'])
        filtered_traffic_data.set_index('FLT_DATE', inplace=True)

        filtered_traffic_data = filtered_traffic_data.resample('D').sum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_traffic_data.index,
            y=filtered_traffic_data['FLT_TOT_1'],
            mode='lines',
            name=selected_airport,
            line=dict(color='blue', dash='solid')
        ))

        stl = STL(filtered_traffic_data['FLT_TOT_1'], seasonal=7, period=7)
        result = stl.fit()

        fig.add_trace(go.Scatter(
            x=result.trend.index,
            y=result.trend,
            mode='lines',
            name='Trend',
            line=dict(color='orange', dash='solid')
        ))
        fig.add_trace(go.Scatter(
            x=result.seasonal.index,
            y=result.seasonal,
            mode='lines',
            name='Seasonal-Week',
            line=dict(color='purple', dash='solid')
        ))
        fig.add_trace(go.Scatter(
            x=result.resid.index,
            y=result.resid,
            mode='lines',
            name='Residual',
            line=dict(color='#42C7DD', dash='solid')
        ))
        fig.update_layout(
            title=f"Daily flight numbers for {selected_airport} over the year {selected_year} with calculation of the Trend, Weekly Seasonality and Residuals",
            xaxis_title="Date",
            template='plotly_white',
        )
    else:
        if selected_country:
            filtered_traffic_data = flight_traffic_data_for_country(traffic_europe, selected_year, selected_country)
        else:
            filtered_traffic_data = top_5_airports_data_for_line_chart(traffic_europe, selected_year, country=None)

        if button_id == 'restyle-button-heatmap':
            pivot_table = filtered_traffic_data.pivot_table(
                values='FLT_TOT_1',
                index='FLT_DATE',
                columns='APT_NAME',
                aggfunc='sum'
            )

            fig = px.imshow(
                pivot_table.T,
                labels=dict(x="Date", y="Airport", color="Total flights"),
                x=pivot_table.index,
                y=pivot_table.columns,
                aspect="auto",
                color_continuous_scale="YlGnBu"
            )
            fig.update_layout(
                template='plotly_white',
                height=340,
                width=1200,
            )
        else:
            fig = px.line(
                filtered_traffic_data,
                x='FLT_DATE',
                y='FLT_TOT_1',
                color='APT_NAME',
                color_discrete_sequence=px.colors.sequential.YlGnBu_r,
                labels={'FLT_TOT_1': 'Total flights', 'FLT_DATE': "Date",'APT_NAME': 'Airport'}
            )
            fig.update_layout(
                template='plotly_white',
                height=340,
                width=1200,
            )
    return fig

@app.callback(
    Output('heatmap', 'children'),
    [Input('country-dropdown', 'value'),
     Input('year-dropdown', 'value'),
     Input('airport-dropdown', "value")]
)
def update_heatmap(selected_country, selected_year, selected_airport):
    if selected_airport:
        heatmap_pivot = generate_heatmap_data(traffic_europe, selected_airport, selected_year)
        fig = px.imshow(
            heatmap_pivot,
            labels=dict(x="Month", y="Weekday", color=" Flights"),
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            color_continuous_scale=px.colors.sequential.ice_r,
            title=f'Average flights per Weekday<br>for {selected_airport} in {selected_year}',
        )
        fig.update_layout(
            height=400,
            width=400,
            margin=dict(l=50, r=50, t=80, b=50),
            template='plotly_white',
            title={
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 16},
            },
        )
        return dcc.Graph(figure=fig)


    elif selected_country:
        country_data = traffic_europe[(traffic_europe['STATE_NAME'] == selected_country) &
                                      (traffic_europe['YEAR'] == selected_year)]
        country_data['WEEKDAY'] = country_data['FLT_DATE'].dt.day_name()
        country_data['MONTH'] = country_data['FLT_DATE'].dt.month
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        country_data['WEEKDAY'] = pd.Categorical(country_data['WEEKDAY'], categories=weekday_order, ordered=True)

        # Berechne die Gesamtanzahl der Flüge pro Flughafen
        total_flights_per_airport = country_data.groupby('APT_NAME')['FLT_TOT_1'].sum().reset_index()

        # Wähle die Top 5 Flughäfen basierend auf der Gesamtanzahl der Flüge
        top_5_airports = total_flights_per_airport.nlargest(5, 'FLT_TOT_1')['APT_NAME']

        # Filtere die Daten, um nur die Top 5 Flughäfen zu enthalten
        filtered_data = country_data[country_data['APT_NAME'].isin(top_5_airports)]

        average_flights_per_weekday_month = filtered_data.groupby(['APT_NAME', 'WEEKDAY', 'MONTH']).agg(
            {'FLT_TOT_1': 'mean'}).reset_index()
        pivot_data = average_flights_per_weekday_month.pivot_table(
            index=['APT_NAME', 'WEEKDAY'],
            columns='MONTH',
            values='FLT_TOT_1'
        ).fillna(0)

        fig = px.imshow(
            pivot_data,
            labels=dict(x="Month", y="Weekday"),  # Set y-axis label to empty
            x=pivot_data.columns,
            y=[f"{index[0]} - {index[1]}" for index in pivot_data.index],
            color_continuous_scale=px.colors.sequential.ice_r,
            aspect="auto",
            title=f'Average Flights per Weekday<br>for Top 5 Airports in {selected_country} in {selected_year}',
        )

        fig.update_layout(
            yaxis=dict(
                showticklabels=False  # Hide y-axis tick labels
            ),
            height=400,
            width=400,  # Adjusted width to accommodate the longer x-axis labels
            margin=dict(l=50, r=50, t=80, b=50),
            template='plotly_white',
            showlegend=False,
            title={
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 16},
            },
        )
        return dcc.Graph(figure=fig)

    else:
        top_5_airports = top_5_airports_by_year(traffic_europe, selected_year)
        mse_matrix = pd.DataFrame(index=top_5_airports['APT_NAME'], columns=top_5_airports['APT_NAME'])

        for airport1 in top_5_airports['APT_NAME']:
            for airport2 in top_5_airports['APT_NAME']:
                if airport1 != airport2:
                    traffic_1 = traffic_europe[traffic_europe["APT_NAME"] == airport1]
                    traffic_2 = traffic_europe[traffic_europe["APT_NAME"] == airport2]
                    mse_tot = calculate_mse(traffic_1, traffic_2, "FLT_TOT_1_norm")
                    mse_matrix.loc[airport1, airport2] = mse_tot
                else:
                    mse_matrix.loc[airport1, airport2] = 0

        mse_matrix = mse_matrix.astype(float)

        fig = go.Figure(data=go.Heatmap(
            z=mse_matrix.values,
            x=mse_matrix.columns,
            y=mse_matrix.index,
            colorscale=px.colors.sequential.ice,
            hoverongaps=False,
        ))
        fig.update_layout(
            height=400,
            width=400,
            margin=dict(l=50, r=50, t=80, b=50),
            template='plotly_white',
            title='How similar Top 5 Airports to each other',
            title_x=0.5,
            title_xanchor='center'
        )
        return dcc.Graph(figure=fig)

@app.callback(
    [Output('country-dropdown', 'value'),
     Output('airport-dropdown', 'value'),
     Output('airline-dropdown', 'value')],
    [Input('reset-button', 'n_clicks')],
    [State('country-dropdown', 'value'),
     State('airport-dropdown', 'value'),
     State('airline-dropdown', 'value')]
)
def reset_filters(n_clicks, country_value, airport_value, airline_value):
    if n_clicks > 0:
        return None, None, None
    else:
        return country_value, airport_value, airline_value

@app.callback(
    Output('airport-dropdown', 'options'),
    [Input('country-dropdown', 'value')]
)
def set_airport_options(selected_country):
    if selected_country is None:
        airports = traffic_europe["APT_NAME"].unique().tolist()
    else:
        airports = traffic_europe[traffic_europe["STATE_NAME"] == selected_country]["APT_NAME"].unique().tolist()

    return [{'label': airport, 'value': airport} for airport in sorted(airports)]


@app.callback(
    Output('airport-info', 'children'),
    Output('airport-info', 'style'),
    [Input('airport-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def update_airport_info(selected_airport, selected_year):
    if selected_airport:
        airport_data = traffic_europe[traffic_europe['APT_NAME'] == selected_airport]
        current_year_flights = airport_data[airport_data['YEAR'] == selected_year]['FLT_TOT_1'].sum()
        previous_year_flights = airport_data[airport_data['YEAR'] == (selected_year - 1)]['FLT_TOT_1'].sum()
        if previous_year_flights > 0:
            flight_comparison = ((current_year_flights - previous_year_flights) / previous_year_flights) * 100
        else:
            flight_comparison = 0

        color = '#39FF14' if flight_comparison >= 0 else 'red'
        arrow = '↑' if flight_comparison >= 0 else '↓'

        daily_flights = airport_data[airport_data['YEAR'] == selected_year].groupby('FLT_DATE')['FLT_TOT_1'].sum()
        popular_day = daily_flights.idxmax()
        max_flights = daily_flights.max()

        return html.Div([
            html.Div(f"Airport KPIs",
                     style={'background-color': '#e7e9eb', 'font-family': 'Open Sans', 'font-size': '20px',
                            'font-weight': 'bold',
                            'margin': '8px'}),
            html.Div(f"Total Number of Flights in {selected_year}: {current_year_flights}",
                     style={'background-color': '#08519C', 'font-family': 'Open Sans', 'font-size': '16px',
                            'font-weight': 'bold', 'margin': '3px', 'color': '#FFFFFF', 'border-radius': '10px'}),
            html.Div([
                f"Comparison to {selected_year - 1}:   ",
                html.Span(f"{arrow} {flight_comparison:.2f}%", style={'color': color}),
            ], style={'background-color': '#08519C', 'font-family': 'Open Sans', 'font-size': '16px',
                      'font-weight': 'bold', 'margin': '3px', 'color': '#FFFFFF', 'border-radius': '10px'}),
            html.Div(
                f"Most popular Day: {popular_day.strftime('%d-%m-%Y')} with {max_flights} flights",
                style={'background-color': '#08519C', 'font-family': 'Open Sans', 'font-size': '16px',
                       'font-weight': 'bold', 'margin': '3px', 'color': '#FFFFFF', 'border-radius': '10px'}),
        ]), {'background-color': '#e7e9eb', 'color': '#000000', 'padding': '10px', 'border-radius': '10px',
             'margin-left': '5px', 'margin-right': '5px', 'display': 'block'}

    else:
        return "", {'display': 'none'}

@app.callback(
    Output('realtime_airport-info', 'children'),
    Output('realtime_airport-info', 'style'),
    [Input('airport-dropdown', 'value')]
)
def update_realtime_airport_info(selected_airport):
    if selected_airport:
        # Get the ICAO code for the selected airport
        airport_icao = traffic_europe.loc[traffic_europe['APT_NAME'] == selected_airport, 'APT_ICAO'].values[0]
        print(f"Selected Airport: {selected_airport}, ICAO: {airport_icao}")  # Debug statement

        flights_df = get_flight_info(airport_icao)
        print(f"Flights DataFrame:\n{flights_df}")  # Debug statement

        if flights_df.empty:
            print("No flights found.")  # Debug statement
            return html.Div([
                html.Div(f"Airport Realtime Flights",
                         style={'font-family': 'Open Sans', 'font-size': '20px', 'font-weight': 'bold',
                                'margin': '10px'}),
                html.Div(f"No real-time flights available for {selected_airport}",
                         style={'font-family': 'Open Sans', 'font-size': '16px', 'font-weight': 'bold', 'margin': '10px'})
            ]), {'background-color': '#b5d3f1', 'color': '#000000', 'padding': '10px', 'border-radius': '10px',
                 'margin-left': '5px', 'margin-right': '5px', 'display': 'block'}
        else:
            print(f"Number of flights: {len(flights_df)}")  # Debug statement

        table = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in flights_df.columns],
            data=flights_df.to_dict('records'),
            style_table={'overflowX': 'scroll'},
            style_cell={'textAlign': 'left'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        )
        return html.Div([
            html.Div(f"Airport Realtime Flights",
                     style={'font-family': 'Open Sans', 'font-size': '20px', 'font-weight': 'bold',
                            'margin': '10px'}),
            html.Div(f"Total Number of Flights for {selected_airport}: {len(flights_df)}",
                     style={'font-family': 'Open Sans', 'font-size': '16px', 'font-weight': 'bold', 'margin': '10px'}),
            table
        ]), {'background-color': '#b5d3f1', 'color': '#000000', 'padding': '10px', 'border-radius': '10px',
             'margin-left': '5px', 'margin-right': '5px', 'display': 'block'}
    else:
        return "", {'display': 'none'}
#_______________________________________________________________________________________________________________________
#Tab 2
@app.callback(
    Output('map1', 'srcDoc'),
    [Input('airline-dropdown', 'value')]
)
def update_airline_map(selected_airline):
    if selected_airline:
        map_object = create_map_with_airlines(airports_location, airlines_routes_data, selected_airline)
        return map_object._repr_html_()
    return create_map_with_airports_2(airports_location)._repr_html_()

@app.callback(
    Output('bar-chart-largest-airports', 'figure'),
    [Input('airline-dropdown', 'value')]
)
def update_bar_chart_hub(selected_airline):
    if selected_airline:
        from_counts = airlines_routes_data[airlines_routes_data['Airline'] == selected_airline]['From'].value_counts().reset_index()
        from_counts.columns = ['Airport', 'Count']
        top_7_hubs = from_counts.head(7)
        fig = px.bar(
            top_7_hubs,
            x='Airport',
            y='Count',
            title=f'Top 7 Hubs for {selected_airline}',
            labels={'Airport': 'Airport', 'Count': 'Number of Flights'},
            color='Airport',
            color_discrete_sequence=px.colors.sequential.YlGnBu_r,
            template='plotly_white',
        )
        fig.update_layout(
            showlegend=False,
            height=290,
            width=400,
            title={
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 16}
            }, margin={'t': 60, "b": 50, "l": 50, "r": 50}
        )
        return fig
    return px.bar(title="Select an airline", template='plotly_white')

@app.callback(
    Output('treemap', 'figure'),
    [Input('airline-dropdown', 'value')]
)
def update_treemap(selected_airline):
    if selected_airline:
        airline_data = airlines_routes_data[airlines_routes_data['Airline'] == selected_airline]
        plane_counts = airline_data['Plane'].value_counts().reset_index()
        plane_counts.columns = ['Plane', 'Count']
        top_15 = plane_counts.head(15)
        fig = px.treemap(
            top_15,
            path=['Plane'],
            values='Count',
            color='Count',
            color_continuous_scale='ice_r',
            title=f'Top 15 Aircraft Types Used<br>by {selected_airline}'
        )
        fig.update_layout(
            template='plotly_white',
            height=290,
            width=400,
            title={
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 16}
            },
            margin={'t': 60, "b": 50, "l": 50, "r": 50}
        )
        return fig
    return px.treemap(title="Select an airline", template='plotly_white')

@app.callback(
    Output('pie', 'figure'),
    [Input('airline-dropdown', 'value')]
)
def update_pie(selected_airline):
    if selected_airline:
        counts = data_pie_chart(selected_airline)
        counts['Type'] = counts['Type'].replace({'Direct': 'Direct', 'Codeshare': 'Codeshare'})

        fig = px.pie(
            counts,
            values='Count',
            color="Type",
            names='Type',
            title=f'Flight Types for {selected_airline}',
            color_discrete_map={'Direct': 'darkblue', 'Codeshare': 'cyan'},
        )
        fig.update_layout(
                template='plotly_white',
                height=290,
                width=400,
                title={
                    'y': 0.9,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 16}
                }, margin={'t': 60, "b": 50, "l": 50, "r": 50}
            )
        return fig
    return px.treemap(title="Select an airline", template='plotly_white')

if __name__ == '__main__':
    app.run_server(debug=True, port=8080)

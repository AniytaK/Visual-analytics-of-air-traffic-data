import pandas as pd
import numpy as np

import folium
import folium.plugins as plugins

from sklearn.metrics import mean_squared_error
from flightradar24 import FlightRadar24API

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

# Functions

def top_5_airports_by_year(traffic_europe, year, country=None):
    if year < 2016 or year > 2024:
        raise ValueError("Year must be between 2016 and 2024")
    traffic_europe['YEAR'] = traffic_europe['YEAR'].astype(int)
    traffic_europe['FLT_DATE'] = pd.to_datetime(traffic_europe['FLT_DATE'])
    filtered_traffic = traffic_europe[traffic_europe['YEAR'] == year]
    if country:
        filtered_traffic = filtered_traffic[filtered_traffic['STATE_NAME'] == country]
    summed_flights = filtered_traffic.groupby(['YEAR', 'APT_NAME'])['FLT_TOT_1'].sum().reset_index()
    average_flights = summed_flights.groupby('APT_NAME')['FLT_TOT_1'].mean().reset_index().sort_values(by='FLT_TOT_1', ascending=False)
    top_5_airports = average_flights.head(5)
    return top_5_airports


def get_bezier_curve_points(start_lat_lon, end_lat_lon, control_point_position=0.5, control_point_offset=0.2, points=100):
    def bezier_curve(p0, p1, p2, n_points=100):
        t = np.linspace(0, 1, n_points)
        points = np.zeros((len(t), 2))
        for i, _t in enumerate(t):
            points[i] = (1 - _t) ** 2 * p0 + 2 * (1 - _t) * _t * p1 + _t ** 2 * p2
        return points.tolist()

    start_lat_lon = np.array(start_lat_lon)
    end_lat_lon = np.array(end_lat_lon)
    direction_vector = end_lat_lon - start_lat_lon
    control_point = start_lat_lon + direction_vector * control_point_position

    if control_point_offset > 0:
        control_point += np.array([-direction_vector[1], direction_vector[0]]) * control_point_offset
    else:
        control_point += np.array([direction_vector[1], -direction_vector[0]]) * abs(control_point_offset)

    curve_points = bezier_curve(start_lat_lon, control_point, end_lat_lon, points)
    return curve_points

def create_map_with_airports(airports_location, airport_routes, selected_country, selected_airport):
    large_airports = airports_location[airports_location['type'] == 'large_airport']
    map_object = folium.Map(location=[50, 24], zoom_start=3)

    for index, row in large_airports.iterrows():
        popup_text = f"Country: {row['country_name']}<br>Airport: {row['name']}"
        popup = folium.Popup(popup_text, max_width=300)

        marker = folium.Marker(
            location=(row['latitude_deg'], row['longitude_deg']),
            icon=plugins.BeautifyIcon(icon="plane", border_color="cadetblue", background_color='white',
                                      icon_color='#304359', icon_size=[15, 15]),
            popup=popup
        )
        marker.add_to(map_object)

    if selected_country:
        country_airports = large_airports[large_airports['country_name'] == selected_country]
        for index, row in country_airports.iterrows():
            popup_text = f"Country: {row['country_name']}<br>Airport: {row['name']}"
            popup = folium.Popup(popup_text, max_width=300)

            marker = folium.Marker(
                location=(row['latitude_deg'], row['longitude_deg']),
                icon=plugins.BeautifyIcon(icon="plane", border_color="orange", background_color='white', icon_color='#304359'),
                popup=popup
            )
            marker.add_to(map_object)

    if selected_airport:
        airport_data = airports_location[airports_location['Airport'] == selected_airport]
        if not airport_data.empty:
            airport = airport_data.iloc[0]
            airport_marker = folium.Marker(
                location=(airport['latitude_deg'], airport['longitude_deg']),
                popup=folium.Popup(f"{airport['name']} Airport<br>{airport['country_name']}", max_width=300),
                icon=plugins.BeautifyIcon(icon="plane", border_color="red", background_color='white', icon_color='black')
            )
            airport_marker.add_to(map_object)

            routes = airport_routes[airport_routes['Airport'] == selected_airport]
            unique_destinations = routes["To"].unique()

            for destination in unique_destinations:
                destination_airport = airports_location[airports_location['iata_code'] == destination]
                if not destination_airport.empty:
                    dest_coords = destination_airport.iloc[0][['latitude_deg', 'longitude_deg']]
                    curve_points = get_bezier_curve_points(
                        (airport['latitude_deg'], airport['longitude_deg']),
                        (dest_coords['latitude_deg'], dest_coords['longitude_deg']),
                        control_point_position=0.5, control_point_offset=0.2
                    )

                    distance = routes[routes['To'] == destination]['Distance'].values[0]
                    from_airport = routes[routes['To'] == destination]['From'].values[0]
                    to_airport = routes[routes['To'] == destination]['To'].values[0]
                    popup_text = f"From: {from_airport}<br>To: {to_airport}<br>Distance: {distance} km"
                    popup = folium.Popup(popup_text, max_width=300)

                    route_line = folium.PolyLine(
                        curve_points,
                        color='blue',
                        weight=2.5,
                        opacity=0.5,
                        popup=popup
                    )
                    route_line.add_to(map_object)

    return map_object


def create_map_with_airports_2(airports_location):
    large_airports = airports_location[airports_location['type'] == 'large_airport']
    map_object = folium.Map(location=[10, 10], zoom_start=3)

    for index, row in large_airports.iterrows():
        popup_text = f"Country: {row['country_name']}<br>Airport: {row['name']}"
        popup = folium.Popup(popup_text, max_width=300)
        marker = folium.Marker(
            location=(row['latitude_deg'], row['longitude_deg']),
            icon=plugins.BeautifyIcon(
                icon="plane", border_color="cadetblue", background_color='white',
                icon_color='#304359', icon_size=[15, 15]
            ),
            popup=popup
        )
        marker.add_to(map_object)
    return map_object

def create_map_with_airlines(airports_location, airlines_routes_data, selected_airline):
    map_object = create_map_with_airports_2(airports_location)

    airline_routes = airlines_routes_data[airlines_routes_data['Airline'] == selected_airline][['From', 'To', 'Distance', 'Duration']].dropna()

    routes_with_coords = airline_routes.merge(airports_location[['iata_code', 'latitude_deg', 'longitude_deg']],
                                              how='left', left_on='From', right_on='iata_code')
    routes_with_coords = routes_with_coords.merge(airports_location[['iata_code', 'latitude_deg', 'longitude_deg']],
                                                  how='left', left_on='To', right_on='iata_code',
                                                  suffixes=('_from', '_to'))

    for index, row in routes_with_coords.iterrows():
        if pd.notna(row['latitude_deg_from']) and pd.notna(row['latitude_deg_to']):
            curve_points = get_bezier_curve_points(
                (row['latitude_deg_from'], row['longitude_deg_from']),
                (row['latitude_deg_to'], row['longitude_deg_to']),
                control_point_position=0.5, control_point_offset=0.2
            )
            popup_text = f"From: {row['From']}<br>To: {row['To']}<br>Distance: {row['Distance']} km<br>Duration: {row['Duration']}"
            route_line = folium.PolyLine(
                locations=curve_points,
                color='blue', weight=1.5, opacity=0.4,
                popup=folium.Popup(popup_text, max_width=300)
            )
            route_line.add_to(map_object)

    from_counts = airline_routes['From'].value_counts().reset_index()
    from_counts.columns = ['Airport', 'Count']

    from_coords = from_counts.merge(airports_location[['iata_code', 'latitude_deg', 'longitude_deg']], how='left',
                                    left_on='Airport', right_on='iata_code')

    for index, row in from_coords.iterrows():
        if pd.notna(row['latitude_deg']) and pd.notna(row['longitude_deg']):
            folium.Circle(
                location=(row['latitude_deg'], row['longitude_deg']),
                radius=row['Count'] * 2000,
                color='orange',
                fill=True,
                fill_color='yellow',
                fill_opacity=0.4,
                popup=f"Airport: {row['Airport']}<br>Routes: {row['Count']}"
            ).add_to(map_object)
    return map_object

def flight_traffic_data_for_country (traffic_europe, year, country_name=None):
    if year < 2016 or year > 2024:
        raise ValueError("Year must be between 2016 and 2024")

    traffic_europe['YEAR'] = traffic_europe['YEAR'].astype(int)
    traffic_europe['FLT_DATE'] = pd.to_datetime(traffic_europe['FLT_DATE'])

    if country_name is None:
        country_name = top_5_airports_by_year()

    country_data = traffic_europe.loc[traffic_europe["STATE_NAME"] == country_name]
    filtered_traffic_data = country_data
    return filtered_traffic_data


def normalize_series(series):
    return (series - series.min()) / (series.max() - series.min()) * 100
traffic_europe['FLT_TOT_1_norm'] = traffic_europe.groupby('APT_NAME')['FLT_TOT_1'].transform(normalize_series)
traffic_europe = traffic_europe.dropna(subset=['FLT_TOT_1_norm'])

def calculate_mse(df1, df2, column):
    df_merged = pd.merge(df1[['FLT_DATE', column]], df2[['FLT_DATE', column]], on='FLT_DATE',
                         suffixes=('_1', '_2'))
    if df_merged.empty:
        return np.nan
    return mean_squared_error(df_merged[f"{column}_1"], df_merged[f"{column}_2"])

def get_most_similar_airports(traffic_europe, selected_airport, column='FLT_TOT_1_norm'):
    # Normalisieren auf einen Wert zwischen 0 und 100%
    def normalize_series(series):
        return (series - series.min()) / (series.max() - series.min()) * 100

    # Normierte Spalten erstellen, NaN-Werte entfernen
    traffic_europe[column] = traffic_europe.groupby('APT_NAME')['FLT_TOT_1'].transform(normalize_series)
    traffic_europe = traffic_europe.dropna(subset=[column])

    base_traffic = traffic_europe.loc[traffic_europe["APT_NAME"] == selected_airport]
    all_airports = traffic_europe["APT_NAME"].unique()

    # Berechnung MSE
    def calculate_mse(df1, df2, column):
        df_merged = pd.merge(df1[['FLT_DATE', column]], df2[['FLT_DATE', column]], on='FLT_DATE',
                             suffixes=('_base', '_other'))
        df_merged = df_merged.dropna()
        return mean_squared_error(df_merged[f"{column}_base"], df_merged[f"{column}_other"])

    # MSE
    distances = []
    for airport in all_airports:
        if airport != selected_airport:
            other_traffic = traffic_europe[traffic_europe["APT_NAME"] == airport]
            mse_tot = calculate_mse(base_traffic, other_traffic, column)
            distances.append((airport, mse_tot))
    distances_df = pd.DataFrame(distances, columns=["Airport", "MSE_TOT"])
    distances_df = distances_df.sort_values(by='MSE_TOT', ascending=True).head(7)
    return distances_df

def generate_heatmap_data(traffic_europe, selected_airport, selected_year):
    df_airport = traffic_europe.loc[(traffic_europe["APT_NAME"] == selected_airport) &
                                               (traffic_europe['FLT_DATE'].dt.year == selected_year)]
    df_airport['Monat'] = df_airport['FLT_DATE'].dt.month
    df_airport['Wochentag'] = df_airport['FLT_DATE'].dt.day_name()
    heatmap_data = df_airport.groupby(['Monat', 'Wochentag'])['FLT_TOT_1'].mean().reset_index()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data['Wochentag'] = pd.Categorical(heatmap_data['Wochentag'], categories=weekday_order, ordered=True)
    heatmap_data = heatmap_data.sort_values('Wochentag')
    heatmap_pivot = heatmap_data.pivot(index='Wochentag', columns='Monat', values='FLT_TOT_1')
    return heatmap_pivot


def get_flight_info(airport_icao):
    fr_api = FlightRadar24API()
    airport_details = fr_api.get_airport_details(airport_icao)

    flights_data = []
    for flight in airport_details.get('flights', []):
        if isinstance(flight, dict):
            flight_id = flight['identification']['id']
            origin_iata = flight['airport']['origin']['code']['iata']
            origin_name = flight['airport']['origin']['name']
            destination_timezone = flight['airport']['destination']['timezone']['name']
            flights_data.append({
                'flight': flight_id,
                'origin': origin_iata,
                'origin_name': origin_name,
                'destination': destination_timezone
            })
    flights_df = pd.DataFrame(flights_data)
    return flights_df

def data_pie_chart(selected_airline):
    direct_count = 0
    codeshare_count = 0

    for i in range(len(airlines_routes_data) - 1):
        if selected_airline in str(airlines_routes_data.iloc[i]['Airline']):
            next_flight_number = airlines_routes_data.iloc[i + 1]['Flight_Number']
            if 'Direct' in next_flight_number:
                direct_count += 1
            elif 'Codeshare' in next_flight_number:
                codeshare_count += 1

    counts = pd.DataFrame({
        'Type': ['Direct', 'Codeshare'],
        'Count': [direct_count, codeshare_count]
    })
    return counts



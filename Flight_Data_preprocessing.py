import pandas as pd

#Combine multiple CSV files-into a single DataFrame to get Flighttraffic (DEP/ARR) about Europe for 2018-2024
def combine_airport_traffic_data():
    data_files = ["airport_traffic_2016.csv",
                  "airport_traffic_2017.csv",
                  "airport_traffic_2018.csv",
                  "airport_traffic_2019.csv",
                  "airport_traffic_2020.csv",
                  "airport_traffic_2021.csv",
                  "airport_traffic_2022.csv",
                  "airport_traffic_2023.csv",
                  "airport_traffic_2024.csv"]

    dataframes = []

    for datei in data_files:
        jahr = datei.split('_')[2].split('.')[0]
        df = pd.read_csv(datei)
        df['Jahr'] = jahr
        dataframes.append(df)

    airport_traffic = pd.concat(dataframes, ignore_index=True)

    return airport_traffic

traffic_europe = combine_airport_traffic_data()


#Get Top 5 Airports per year
def top_5_airports_by_year_for_barchart(traffic_europe, year, country=None):
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

def top_5_airports_data_for_line_chart(traffic_europe, year, country=None):

    top_5_airports = top_5_airports_by_year_for_barchart(traffic_europe, year, country)
    top_5_airports_names = top_5_airports['APT_NAME'].tolist()
    filtered_data = traffic_europe[
        (traffic_europe['APT_NAME'].isin(top_5_airports_names)) &
        (traffic_europe['YEAR'].between(2016, 2024))
    ]
    return filtered_data

#Get data for specific country

def get_country_data(traffic_europe, country_name):
    country_data = traffic_europe.loc[traffic_europe["STATE_NAME"] == country_name]
    return country_data

#
def airport_location_coordinates():
    location = pd.read_csv("world-airports.csv")
    traffic_europe = combine_airport_traffic_data()
    iata_to_airport_name = traffic_europe.set_index('APT_ICAO')['APT_NAME'].to_dict()

    def map_iata_to_airport(iata_code):
        return iata_to_airport_name.get(iata_code, 'Unknown Airport')
    location['Airport'] = location['ident'].map(map_iata_to_airport)
    return location

def combine_airport_routes_data():
    data_files = [
        "Adolfo Suárez Madrid–Barajas Airport.csv",
        "Amsterdam Airport Schiphol.csv",
        "Atatürk International Airport.csv",
        "Barcelona International Airport.csv",
        "Charles de Gaulle International Airport.csv",
        "Zurich.csv",
        "Stuttgart Airport .csv",
        "Munich Airport .csv",
        "London Heathrow Airport .csv",
        "London Gatwick Airport .csv",
        "Leonardo da Vinci–Fiumicino Airport.csv",
        "Gran Canaria Airport.csv",
        "Frankfurt_am_Mein_Airport.csv",
        "Marseille Provence Airport.csv",
        "Tirana International Airport.csv",
        "Graz Airport.csv",
        "Innsbruck Airport.csv",
        "Linz Hörsching Airport .csv",
        "Klagenfurt Airport.csv",
        "Salzburg Airport.csv",
        "Antalya International Airport (AYT).csv"
    ]
    dataframes = []
    for datei in data_files:
        df = pd.read_csv(datei)
        dataframes.append(df)

    airport_routes = pd.concat(dataframes, ignore_index=True)
    iata_to_airport_name = {
        'MAD': 'Madrid - Barajas',
        'AMS': 'Amsterdam - Schiphol',
        'ISL': 'iGA Istanbul Airport',
        'BCN': 'Barcelona',
        'CDG': 'Paris-Charles-de-Gaulle',
        "ZRH": 'Zürich',
        "STR": 'Stuttgart',
        "MUC": 'Munich',
        "LHR": 'London - Heathrow',
        "LGW": 'London - Gatwick',
        "FCO": 'Rome - Fiumicino',
        "LPA": 'Gran Canaria',
        "FRA": 'Frankfurt',
        "MRS": 'Marseille-Provence',
        "TIA": 'Tirana',
        "GRZ": 'Graz',
        "INN": 'Innsbruck',
        "LNZ": 'Linz',
        "KLU": 'Klagenfurt',
        "SZG": 'Salzburg',
        "AYT": "Antalya"
    }
    def map_iata_to_airport(iata_code):
        return iata_to_airport_name.get(iata_code, 'Unknown Airport')
    airport_routes['Airport'] = airport_routes['From'].map(map_iata_to_airport)

    return airport_routes

#Horisontal flight data
def combine_flight_efficiency_data():
    file_paths = [
        "horizontal_flight_efficiency_2018.csv",
        "horizontal_flight_efficiency_2019.csv",
        "horizontal_flight_efficiency_2020.csv",
        "horizontal_flight_efficiency_2021.csv",
        "horizontal_flight_efficiency_2022.csv",
        "horizontal_flight_efficiency_2023.csv",
        "horizontal_flight_efficiency_2024.csv"
    ]
    dataframes = []

    for file in file_paths:
        jahr = file.split('_')[2].split('.')[0]
        df = pd.read_csv(file)
        df['Jahr'] = jahr
        dataframes.append(df)
    horisontal_efficiency = pd.concat(dataframes, ignore_index=True)
    return horisontal_efficiency
horisontal_flight_eff = combine_flight_efficiency_data()

def combine_airlines_routes_data():
    data_files = ["Air Berlin.csv",
                  "Air France.csv",
                  "Austrian Airlines.csv",
                  "Delta Air Lines (DL).csv",
                  "Ethiopian Airlines.csv",
                  "Germanwings.csv",
                  "Helvetic Airways.csv",
                  "KLM Royal Dutch Airlines.csv",
                  "Lufthansa.csv",
                  "Swiss International Air Lines.csv",
                  "United Airlines.csv"]
    dataframes = []
    for datei in data_files:
        df = pd.read_csv(datei)
        dataframes.append(df)
    airlines_routes = pd.concat(dataframes, ignore_index=True)
    return airlines_routes
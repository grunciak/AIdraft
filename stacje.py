import streamlit as st
import requests
import pandas as pd

# Endpointy API
API_BASE = 'http://vps.atmesys.com/api/v4/data/a8f5f167f44f4964e6c998dee827110c'
LIST_ENDPOINT = f'{API_BASE}/list'

def get_station_keys():
    response = requests.get(LIST_ENDPOINT)
    if response.status_code == 200:
        data = response.json()
        return data['station_keys']
    else:
        st.error('Nie udało się pobrać listy stacji.')
        return []

def get_station_data(station_key):
    endpoint = f'{API_BASE}/{station_key}/last'
    response = requests.get(endpoint)
    if response.status_code == 200:
        return response.json()
    else:
        st.warning(f'Nie udało się pobrać danych dla stacji {station_key}.')
        return None

def main():
    st.title('Dane ze stacji pomiarowych')
    station_keys = get_station_keys()
    
    if station_keys:
        for station_key in station_keys:
            data = get_station_data(station_key)
            if data:
                st.subheader(f'Stacja: {data.get("station_name", "N/A")}')
                df = pd.DataFrame([data['measurements']])
                st.dataframe(df)
                # Możesz dodać wykresy lub inne elementy graficzne tutaj
    else:
        st.error('Brak dostępnych stacji do wyświetlenia.')

if __name__ == '__main__':
    main()

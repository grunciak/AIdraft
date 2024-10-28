import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import datetime as dt

# Funkcja do wczytywania danych
def load_data(uploaded_file):
    df = pd.read_excel(uploaded_file)
    df['data'] = pd.to_datetime(df['data'])  # konwersja kolumny 'data' na typ datetime
    df['zuzycie'] = pd.to_numeric(df['zuzycie'], errors='coerce')  # konwersja kolumny 'zuzycie' na numeryczną
    df = df.dropna().reset_index(drop=True)
    return df

# Główna funkcja aplikacji
def main():
    st.title('Predykcja zużycia wody')
    st.write('Aplikacja do przewidywania zużycia wody na podstawie wcześniejszych danych.')

    # Wczytanie pliku
    uploaded_file = st.file_uploader("Wybierz plik Excel", type=["xls","xlsx"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        # Wyświetlanie załadowanych danych
        st.write("Wyświetlanie pierwszych 5 wierszy danych:")
        st.write(df.head())

        # Przygotowanie danych do modelu
        df['timestamp'] = df['data'].map(dt.datetime.toordinal)
        X = df['timestamp'].values.reshape(-1,1)
        y = df['zuzycie'].values

        # Podział danych na zestaw treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Trenowanie modelu
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Przewidywania na podstawie modelu
        y_pred = model.predict(X_test)

        # Obliczanie i wyświetlanie błędu średniokwadratowego i współczynnika determinacji R^2
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f'Błąd średniokwadratowy (MSE): {mse:.2f}')
        st.write(f'Współczynnik determinacji (R^2): {r2:.2f}')

        # Wykres danych
        fig, ax = plt.subplots()
        ax.scatter(df['data'], df['zuzycie'], color='black', label='Dane rzeczywiste')
        ax.plot(df['data'], model.predict(df['timestamp'].values.reshape(-1,1)), color='blue', linewidth=3, label='Linia trendu')
        ax.set_xlabel('Data')
        ax.set_ylabel('Zużycie wody')
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend()
        st.pyplot(fig)

        # Wybór daty do predykcji
        selected_date = st.date_input("Wybierz datę do predykcji zużycia wody")
        if st.button('Przewiduj'):
            selected_date = dt.datetime.strptime(str(selected_date), '%Y-%m-%d')
            selected_date_ordinal = selected_date.toordinal()
            predicted_consumption = model.predict([[selected_date_ordinal]])
            st.write(f'Przewidywane zużycie wody na dzień {selected_date.date()}: {predicted_consumption[0]:.2f}')

if __name__ == '__main__':
    main()

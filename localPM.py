import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from xgboost import XGBClassifier
import datetime

# Streamlit interface
st.title('Predykcja Alarmów')
st.write('Dane są automatycznie wczytywane z pliku "2810Merged_PM_Data2.xlsx".')

# Automatyczne wczytanie pliku z lokalnego katalogu
file_path = '2810Merged_PM_Data2.xlsx'
try:
    data = pd.read_excel(file_path)
except FileNotFoundError:
    st.write("Plik '2810Merged_PM_Data2.xlsx' nie został znaleziony w lokalnym katalogu.")
    st.stop()

# Konwersja kolumny 'date' na typ datetime
if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'])
else:
    st.write("Brak kolumny 'date'.")
    st.stop()

# Wyświetlanie pierwszych kilku wierszy i dostępnych kolumn dla weryfikacji
st.write("Podgląd danych:", data.head())
st.write("Dostępne kolumny:", data.columns.tolist())

# Definiowanie kolumn, które nie są alarmami
exclude_columns = [
    'CT-RS485 Temperatura', 'KTW-2 Temperatura', 'KTW-2 Wilgotność', 'hour', 'day_of_week', 'date'
]

# Identyfikacja kolumn alarmowych przez wykluczenie znanych kolumn niealarmowych
alarm_columns = [col for col in data.columns if col not in exclude_columns]

st.write("Zidentyfikowane kolumny alarmów:", alarm_columns)

if alarm_columns:
    # Selectbox do wyboru kolumny alarmu
    selected_alarm = st.selectbox('Wybierz kolumnę alarmu', alarm_columns)

    # Wypełnianie wartości NaN w kolumnie alarmu wartością 0
    data[selected_alarm] = data[selected_alarm].fillna(0)

    # Feature engineering: dodanie kolumn z godziną i dniem tygodnia
    data['hour'] = data['date'].dt.hour
    data['day_of_week'] = data['date'].dt.dayofweek

    # Lista cech do trenowania modelu
    features = [col for col in data.columns if col not in ['date', selected_alarm]]

    # Przygotowanie cech i celu do trenowania modelu
    X = data[features]
    y = data[selected_alarm].apply(lambda x: 0 if x == 0 else 1)  # Binary classification

    # Wyświetlenie unikalnych wartości w celu debugowania potencjalnych problemów
    unique_targets = y.unique()
    st.write("Unikalne wartości w kolumnie docelowej (y):", unique_targets)

    # Sprawdzenie, czy zmienna docelowa jest binarna
    if len(unique_targets) > 2 or not set(unique_targets).issubset({0, 1}):
        st.write(f"Error: Target variable has unexpected values: {unique_targets}")
    else:
        # Podział na zbiór treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Trenowanie modelu
        model = XGBClassifier()
        model.fit(X_train, y_train)

        # Ewaluacja modelu
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)

        # Wyświetlenie metryk jakości predykcji
        st.write('## Miary jakości predykcji')
        st.write(f'Accuracy: {accuracy:.2f}')
        st.write(f'Precyzja: {precision:.2f}')
        st.write(f'Recall: {recall:.2f}')
        st.write(f'F1-score: {f1:.2f}')

        # Wyświetlanie korelacji
        correlations = data[features + [selected_alarm]].corr()
        correlations = correlations.fillna(0)
        st.write(f'## Korelacje cech z kolumną {selected_alarm}')
        st.dataframe(correlations[[selected_alarm]])

        # Wybór daty przez użytkownika, z możliwością przyszłych dat
        max_date_in_data = data['date'].max()
        future_max_date = max_date_in_data + datetime.timedelta(days=365)
        selected_date = st.date_input('Wybierz datę', min_value=max_date_in_data, max_value=future_max_date)

        # Predykcja dla wybranej daty
        def prepare_features(date):
            hour = date.hour
            day_of_week = date.dayofweek

            # Użycie najnowszych dostępnych danych do wypełnienia innych cech
            recent_data = data.tail(1).copy()
            recent_data['hour'] = hour
            recent_data['day_of_week'] = day_of_week

            return recent_data[features]

        if st.button('Sprawdź alarm'):
            input_data = prepare_features(pd.to_datetime(selected_date))
            prediction = model.predict(input_data)
            if prediction[0] == 1:
                st.write(f'Alarm wystąpi {selected_date}.')
            else:
                st.write(f'Brak alarmów {selected_date}.')
else:
    st.write("Brak dostępnych kolumn alarmowych w danych.")


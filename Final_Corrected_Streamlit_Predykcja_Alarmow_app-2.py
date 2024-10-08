
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import datetime

# Streamlit interface
st.title('Predykcja Alarmów')
st.write('Wybierz kolumnę alarmu oraz datę, aby zobaczyć, czy wystąpi alarm.')

# File uploader for monitoring data
monitoring_file = st.file_uploader("Wgraj plik z danymi monitorowania", type=["xlsx"])
alarm_file = st.file_uploader("Wgraj plik z danymi alarmów", type=["xlsx"])

if monitoring_file and alarm_file:
    # Wczytanie danych
    monitoring_data = pd.read_excel(monitoring_file)
    alarm_data = pd.read_excel(alarm_file)

    # Clean column names in alarm_data
    alarm_data.columns = alarm_data.columns.str.strip()

    # Convert 'date' column to datetime
    monitoring_data['date'] = pd.to_datetime(monitoring_data['date'])
    alarm_data['date'] = pd.to_datetime(alarm_data['date'])

    # Merge datasets on the date column
    data = monitoring_data.merge(alarm_data, on='date', how='left')

    # Fill NaN values with 0 for alarm columns
    alarm_columns = [
        'SSP - awaria ogólna', 'SSP - awaria zasilania', 'SSP - pożar I stopnia',
        'SSP - pożar II stopnia', 'UDK - awaria ogólna', 'UDK - awaria zasilania',
        'UDK - sabotaż', 'UDK - włamanie'
    ]
    data[alarm_columns] = data[alarm_columns].fillna(0)

    # Feature engineering
    data['hour'] = data['date'].dt.hour
    data['day_of_week'] = data['date'].dt.dayofweek

    # List of features for model training
    features = data.columns.difference(['date'] + alarm_columns).tolist() + ['hour', 'day_of_week']

    # Selectbox for choosing the alarm column
    selected_alarm = st.selectbox('Wybierz kolumnę alarmu', alarm_columns)

    # Obliczanie najnowszej daty
    latest_date = monitoring_data['date'].max()
    min_date = latest_date.date()
    max_date = min_date + datetime.timedelta(days=14)

    # Wybór daty przez użytkownika
    selected_date = st.date_input('Wybierz datę', min_value=min_date, max_value=max_date)

    # Przygotowanie danych do predykcji
    def prepare_features(date):
        date = pd.to_datetime(date)
        hour = date.hour
        day_of_week = date.dayofweek
        recent_data = monitoring_data[monitoring_data['date'] <= date].tail(1)
        if recent_data.empty:
            return None
        recent_data['hour'] = hour
        recent_data['day_of_week'] = day_of_week
        
        # Debugging: Check columns in recent_data and features
        st.write("Columns in recent_data:", recent_data.columns.tolist())
        st.write("Features:", features)
        
        missing_cols = set(features) - set(recent_data.columns)
        if missing_cols:
            st.write(f"Missing columns: {missing_cols}")
        
        return recent_data[features]

    # Prepare features and target for model training
    X = data[features]
    y = data[selected_alarm]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)

    try:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
    except ValueError as e:
        st.write(f"Error calculating metrics: {e}")
        accuracy = precision = recall = f1 = 0.0

    # Calculate correlations
    correlations = data[features + [selected_alarm]].corr()

    # Predykcja alarmu
    if st.button('Sprawdź alarm'):
        input_data = prepare_features(selected_date)
        if input_data is not None:
            prediction = model.predict(input_data)
            if prediction[0] == 1:
                st.write(f'Alarm wystąpi {selected_date}.')
            else:
                st.write(f'Brak alarmów {selected_date}.')
        else:
            st.write(f'Brak danych do predykcji dla wybranej daty.')

    # Wyświetlanie miar jakości modelu
    st.write('## Miary jakości predykcji')
    st.write(f'Accuracy: {accuracy:.2f}')
    st.write(f'Precyzja: {precision:.2f}')
    st.write(f'Recall: {recall:.2f}')
    st.write(f'F1-score: {f1:.2f}')

    # Wyświetlanie korelacji
    st.write(f'## Korelacje cech z kolumną {selected_alarm}')
    st.write(correlations[[selected_alarm]])

else:
    st.write("Proszę wgrać oba pliki, aby kontynuować.")

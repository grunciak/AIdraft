import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import datetime

# Streamlit interface
st.title('Predykcja Alarmów')
st.write('Dane zostaną automatycznie wczytane.')

# Load local file
file_path = '2810Merged_PM_Data2.xlsx'
try:
    data = pd.read_excel(file_path)
except FileNotFoundError:
    st.write("Plik '2810Merged_PM_Data2.xlsx' nie został znaleziony w lokalnym katalogu.")
    st.stop()

# Data processing
if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'])
else:
    st.write("Brak kolumny 'date'.")
    st.stop()

st.write("Podgląd danych:", data.head())
exclude_columns = ['CT-RS485 Temperatura', 'KTW-2 Temperatura', 'KTW-2 Wilgotność', 'hour', 'day_of_week', 'date']
alarm_columns = [col for col in data.columns if col not in exclude_columns]

st.write("Zidentyfikowane kolumny alarmów:", alarm_columns)

if alarm_columns:
    selected_alarm = st.selectbox('Wybierz kolumnę alarmu', alarm_columns)
    data[selected_alarm] = data[selected_alarm].fillna(0)
    data['hour'] = data['date'].dt.hour
    data['day_of_week'] = data['date'].dt.dayofweek

    features = [col for col in data.columns if col not in ['date', selected_alarm]]

    # Date range selection for prediction
    start_date = st.date_input("Wybierz początkową datę predykcji", data['date'].min().date())
    end_date = st.date_input("Wybierz końcową datę predykcji", data['date'].max().date())

    if start_date > end_date:
        st.error("Data początkowa nie może być późniejsza niż końcowa.")
    else:
        # Filter data based on selected date range
        filtered_data = data[(data['date'] >= pd.to_datetime(start_date)) & (data['date'] <= pd.to_datetime(end_date))]

        st.write("Podgląd przefiltrowanych danych:", filtered_data.head())

        if not filtered_data.empty:
            # Prediction setup
            X = filtered_data[features]
            y = filtered_data[selected_alarm]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Model training
            model = XGBClassifier()
            model.fit(X_train, y_train)

            # Make predictions
            predictions = model.predict(X_test)

            # Display results
            st.write("Wyniki predykcji:")
            st.write("Precyzja:", precision_score(y_test, predictions))
            st.write("Recall:", recall_score(y_test, predictions))
            st.write("F1 Score:", f1_score(y_test, predictions))
            st.write("Dokładność:", accuracy_score(y_test, predictions))
        else:
            st.warning("Brak danych do predykcji w wybranym zakresie dat.")
else:
    st.warning("Brak kolumn alarmów do wyboru.")

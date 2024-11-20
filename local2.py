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

# Load local file with a fixed path
file_path = '2810Merged_PM_Data2.xlsx'
try:
    data = pd.read_excel(file_path)
except FileNotFoundError:
    st.write("Plik '2810Merged_PM_Data2.xlsx' nie został znaleziony w lokalnym katalogu.")
    st.stop()

# Convert 'date' column to datetime
if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'])
else:
    st.write("Brak kolumny 'date'.")
    st.stop()

# Display the first few rows of the dataset for verification
st.write("Podgląd danych:", data.head())

# Define columns to exclude from alarm columns
exclude_columns = [
    'CT-RS485 Temperatura', 'KTW-2 Temperatura', 'KTW-2 Wilgotność', 'hour', 'day_of_week', 'date'
]

# Identify alarm columns by excluding known non-alarm columns
alarm_columns = [col for col in data.columns if col not in exclude_columns]
st.write("Zidentyfikowane kolumny alarmów:", alarm_columns)
 # Diagnostyka - sprawdzenie rozkładu klas przed SMOTE
    class_counts = y_train.value_counts()
    st.write("Rozkład klas w y_train:", class_counts)
if alarm_columns:
    # Selectbox for choosing the alarm column
    selected_alarm = st.selectbox('Wybierz kolumnę alarmu', alarm_columns)
    data[selected_alarm] = data[selected_alarm].fillna(0)
    data['hour'] = data['date'].dt.hour
    data['day_of_week'] = data['date'].dt.dayofweek

    # Features for the model
    features = [col for col in data.columns if col not in ['date', selected_alarm]]

    # Option to handle imbalanced data
    imbalance_option = st.selectbox(
        'Wybierz metodę dla niezbalansowanych danych:',
        ('Bez modyfikacji', 'SMOTE', 'Undersampling', 'SMOTEENN')
    )

    # Prepare data for model training
    X = data[features]
    y = data[selected_alarm]

    # Apply chosen imbalance handling method
    if imbalance_option == 'SMOTE':
        X, y = SMOTE().fit_resample(X, y)
    elif imbalance_option == 'Undersampling':
        X, y = RandomUnderSampler().fit_resample(X, y)
    elif imbalance_option == 'SMOTEENN':
        X, y = SMOTEENN().fit_resample(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # Predictions for evaluation
    y_pred = model.predict(X_test)

    # Display model evaluation metrics
    st.write("Miary jakości modelu:")
    st.write("Precyzja:", precision_score(y_test, y_pred))
    st.write("Recall:", recall_score(y_test, y_pred))
    st.write("F1 Score:", f1_score(y_test, y_pred))
    st.write("Dokładność:", accuracy_score(y_test, y_pred))

    # Future date input for single-day prediction
    max_date_in_data = data['date'].max()
    future_max_date = max_date_in_data + datetime.timedelta(days=365)
    selected_date = st.date_input(
        'Wybierz datę do przewidzenia alarmu', 
        min_value=max_date_in_data, 
        max_value=future_max_date
    )
 # Analiza korelacji
    st.write("## Korelacja cech z wybraną kolumną alarmową")
    correlation_matrix = data[features + [selected_alarm]].corr()
    st.write("Korelacje między cechami a kolumną alarmową:")
    st.dataframe(correlation_matrix[[selected_alarm]].sort_values(by=selected_alarm, ascending=False))
    # Function to prepare features for the selected date
    def prepare_features(date):
        hour = date.hour
        day_of_week = date.weekday()
        
        # Use the most recent available data to fill other features
        recent_data = data.tail(1).copy()
        recent_data['hour'] = hour
        recent_data['day_of_week'] = day_of_week

        return recent_data[features]

    # Prediction for the selected future date
    if st.button('Przewidź alarm dla wybranej daty'):
        input_data = prepare_features(pd.to_datetime(selected_date))
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.write(f'Alarm wystąpi {selected_date}.')
        else:
            st.write(f'Brak alarmów {selected_date}.')
else:
    st.warning("Brak kolumn alarmowych do wyboru.")

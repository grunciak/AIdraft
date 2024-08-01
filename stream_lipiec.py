import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from xgboost import XGBClassifier
import datetime

# Streamlit interface
st.title('Predykcja Alarmów')
st.write('Wybierz kolumnę alarmu oraz datę, aby zobaczyć, czy wystąpi alarm.')

# File uploader for monitoring data
monitoring_file = st.file_uploader("Wgraj plik z danymi monitorowania", type=["xlsx"], key="file_uploader_monitoring")
alarm_file = st.file_uploader("Wgraj plik z danymi alarmów", type=["xlsx"], key="file_uploader_alarm")

if monitoring_file and alarm_file:
    # Read the uploaded Excel files
    monitoring_data = pd.read_excel(monitoring_file)
    alarm_data = pd.read_excel(alarm_file)

    # Clean column names
    monitoring_data.columns = monitoring_data.columns.str.strip()
    alarm_data.columns = alarm_data.columns.str.strip()

    # Display columns to debug
    st.write("Columns in monitoring data:", monitoring_data.columns.tolist())
    st.write("Columns in alarm data:", alarm_data.columns.tolist())

    # Convert 'date' column to datetime
    monitoring_data['date'] = pd.to_datetime(monitoring_data['date'])
    alarm_data['date'] = pd.to_datetime(alarm_data['date'])

    # Merge datasets on the date column
    data = pd.merge(monitoring_data, alarm_data, on='date', how='left', suffixes=('_monitor', '_alarm'))

    # Display columns after merge
    st.write("Columns in merged data:", data.columns.tolist())

    # List of alarm columns
    alarm_columns = [
        'SSP - awaria ogólna_alarm', 'SSP - awaria zasilania_alarm', 'SSP - pożar I stopnia_alarm',
        'SSP - pożar II stopnia_alarm', 'UDK - awaria ogólna_alarm', 'UDK - awaria zasilania_alarm',
        'UDK - sabotaż_alarm', 'UDK - włamanie_alarm'
    ]

    # Identify alarm columns present in the data
    existing_alarm_columns = [col for col in alarm_columns if col in data.columns]

    # Check if there are any missing alarm columns
    missing_alarm_columns = set(alarm_columns) - set(existing_alarm_columns)
    if missing_alarm_columns:
        st.write(f"Warning: The following alarm columns were not found in the data: {missing_alarm_columns}")

    if existing_alarm_columns:
        # Fill NaN values in the existing alarm columns with 0
        data[existing_alarm_columns] = data[existing_alarm_columns].fillna(0)

        # Feature engineering: extract hour and day of the week from 'date'
        data['hour'] = data['date'].dt.hour
        data['day_of_week'] = data['date'].dt.dayofweek

        # Separate numeric columns for imputation
        numeric_cols = data.select_dtypes(include=np.number).columns
        non_numeric_cols = data.select_dtypes(exclude=np.number).columns

        # Impute missing values in numeric columns using the mean
        imputer = SimpleImputer(strategy='mean')
        data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

        # Handle missing data in non-numeric columns (forward fill)
        data[non_numeric_cols] = data[non_numeric_cols].fillna(method='ffill')

        # Ensure all numeric columns are properly formatted
        for feature in numeric_cols:
            data[feature] = pd.to_numeric(data[feature], errors='coerce')

        # List of features for model training
        features = data.columns.difference(['date'] + existing_alarm_columns).tolist() + ['hour', 'day_of_week']

        # Debugging: Print data types and check for NaNs
        st.write("Data types of features:", data[features].dtypes)
        st.write("Check for NaNs in features:", data[features].isnull().sum())
        st.write("Unique values in target variable (y):", data[existing_alarm_columns].apply(pd.Series.nunique))

        # Selectbox for choosing the alarm column
        selected_alarm = st.selectbox('Wybierz kolumnę alarmu', existing_alarm_columns, key="selectbox_alarm")

        # Clean target variable (y) to ensure binary classification (0 or 1)
        y = data[selected_alarm].apply(lambda x: 0 if x == 0 else 1)

        # Check that all features are numeric and there are no NaNs
        if data[features].isnull().any().any():
            st.write("Error: NaN values detected in features.")
            st.stop()

        if y.isnull().any():
            st.write("Error: NaN values detected in the target variable.")
            st.stop()

        # Debugging: Print data types of features to ensure they are all numeric
        st.write("Feature data types:", data[features].dtypes)

        # Calculate the latest date available
        latest_date = monitoring_data['date'].max()
        min_date = latest_date.date()
        max_date = min_date + datetime.timedelta(days=14)

        # User date input
        selected_date = st.date_input('Wybierz datę', min_value=min_date, max_value=max_date, key="date_input")

        # Prepare data for prediction
        def prepare_features(date):
            date = pd.to_datetime(date)
            hour = date.hour
            day_of_week = date.dayofweek

            # Get the most recent data up to the selected date
            recent_data = data[data['date'] <= date].tail(1)
            if recent_data.empty:
                return None

            recent_data['hour'] = hour
            recent_data['day_of_week'] = day_of_week

            return recent_data[features]

        # Prepare features and target for model training
        X = data[features]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train the model
        model = XGBClassifier()
        try:
            model.fit(X_train, y_train)
        except ValueError as e:
            st.write(f"Error during model training: {e}")
            st.write("Check data types and NaN values in features and target.")
            st.stop()

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)

        # Display model metrics
        st.write('## Miary jakości predykcji')
        st.write(f'Accuracy: {accuracy:.2f}')
        st.write(f'Precyzja: {precision:.2f}')
        st.write(f'Recall: {recall:.2f}')
        st.write(f'F1-score: {f1:.2f}')

        # Display correlations
        correlations = data[features + [selected_alarm]].corr()
        st.write(f'## Korelacje cech z kolumną {selected_alarm}')
        st.write(correlations[[selected_alarm]])

        # Alarm prediction
        if st.button('Sprawdź alarm', key="button_alarm"):
            input_data = prepare_features(selected_date)
            if input_data is not None:
                prediction = model.predict(input_data)
                if prediction[0] == 1:
                    st.write(f'Alarm wystąpi {selected_date}.')
                else:
                    st.write(f'Brak alarmów {selected_date}.')
            else:
                st.write(f'Brak danych do predykcji dla wybranej daty.')
    else:
        st.write("No alarm columns are available in the uploaded data.")

else:
    st.write("Proszę wgrać oba pliki, aby kontynuować.")

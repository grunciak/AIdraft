import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from xgboost import XGBClassifier

# Streamlit interface
st.title('Predykcja Alarmów z Pojedynczego Pliku')
st.write('Wgraj plik z danymi i wybierz kolumnę alarmu oraz datę, aby zobaczyć, czy wystąpi alarm.')

# File uploader for the merged data
uploaded_file = st.file_uploader("Wgraj plik z danymi", type=["xlsx"])

if uploaded_file:
    # Read the uploaded Excel file
    data = pd.read_excel(uploaded_file)

    # Display all column names
    st.write("Dostępne kolumny:", data.columns.tolist())

    # Check for a 'date' column and convert it to datetime
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
    else:
        st.write("Brak kolumny 'date'.")
        st.stop()

    # Display the first few rows of the dataset for verification
    st.write("Podgląd danych:", data.head())

    # Assume alarm columns are those with numeric data, excluding the 'date' and 'hour', 'day_of_week'
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    alarm_columns = [col for col in numeric_cols if col not in ['hour', 'day_of_week']]

    st.write("Zidentyfikowane kolumny alarmów:", alarm_columns)

    if alarm_columns:
        # Selectbox for choosing the alarm column
        selected_alarm = st.selectbox('Wybierz kolumnę alarmu', alarm_columns)

        # Fill NaN values in the selected alarm column with 0
        data[selected_alarm] = data[selected_alarm].fillna(0)

        # Feature engineering: extract hour and day of the week from 'date'
        data['hour'] = data['date'].dt.hour
        data['day_of_week'] = data['date'].dt.dayofweek

        # List of features for model training
        features = [col for col in data.columns if col not in ['date', selected_alarm]]

        # Prepare features and target for model training
        X = data[features]
        y = data[selected_alarm].apply(lambda x: 0 if x == 0 else 1)  # Ensure binary classification

        # Display unique values in the target to debug potential issues
        unique_targets = y.unique()
        st.write("Unikalne wartości w kolumnie docelowej (y):", unique_targets)

        # Check that the target variable is binary
        if len(unique_targets) > 2 or not set(unique_targets).issubset({0, 1}):
            st.write(f"Error: Target variable has unexpected values: {unique_targets}")
        else:
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Train the model
            model = XGBClassifier()
            model.fit(X_train, y_train)

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
            correlations = correlations.fillna(0)
            st.write(f'## Korelacje cech z kolumną {selected_alarm}')
            st.dataframe(correlations[[selected_alarm]])

            # User date input
            selected_date = st.date_input('Wybierz datę', min_value=data['date'].min().date(), max_value=data['date'].max().date())

            # Prediction for selected date
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
    else:
        st.write("Brak dostępnych kolumn alarmowych w danych.")
else:
    st.write("Proszę wgrać plik z danymi, aby kontynuować.")


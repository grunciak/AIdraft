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
st.title('Predykcja Alarmów z Pojedynczego Pliku')
st.write('Wgraj plik z danymi i wybierz kolumnę alarmu oraz datę, aby zobaczyć, czy wystąpi alarm.')

uploaded_file = st.file_uploader("Wgraj plik z danymi", type=["xlsx"])

if uploaded_file:
    data = pd.read_excel(uploaded_file)
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
        X = data[features]
        y = data[selected_alarm].apply(lambda x: 0 if x == 0 else 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Define sampling methods
        smote = SMOTE(random_state=42)
        undersampler = RandomUnderSampler(random_state=42)
        smoteenn = SMOTEENN(random_state=42)

        # Original model without sampling
        model = XGBClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        original_metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-score": f1_score(y_test, y_pred, zero_division=0)
        }
        st.write("## Wyniki bez balansowania")
        st.write(original_metrics)

        # SMOTE
        X_res, y_res = smote.fit_resample(X_train, y_train)
        model_smote = XGBClassifier()
        model_smote.fit(X_res, y_res)
        y_pred_smote = model_smote.predict(X_test)
        smote_metrics = {
            "Accuracy": accuracy_score(y_test, y_pred_smote),
            "Precision": precision_score(y_test, y_pred_smote, zero_division=0),
            "Recall": recall_score(y_test, y_pred_smote, zero_division=0),
            "F1-score": f1_score(y_test, y_pred_smote, zero_division=0)
        }
        st.write("## Wyniki z SMOTE")
        st.write(smote_metrics)

        # Undersampling
        X_res, y_res = undersampler.fit_resample(X_train, y_train)
        model_undersample = XGBClassifier()
        model_undersample.fit(X_res, y_res)
        y_pred_undersample = model_undersample.predict(X_test)
        undersample_metrics = {
            "Accuracy": accuracy_score(y_test, y_pred_undersample),
            "Precision": precision_score(y_test, y_pred_undersample, zero_division=0),
            "Recall": recall_score(y_test, y_pred_undersample, zero_division=0),
            "F1-score": f1_score(y_test, y_pred_undersample, zero_division=0)
        }
        st.write("## Wyniki z undersamplingiem")
        st.write(undersample_metrics)

        # SMOTE + ENN
        X_res, y_res = smoteenn.fit_resample(X_train, y_train)
        model_smoteenn = XGBClassifier()
        model_smoteenn.fit(X_res, y_res)
        y_pred_smoteenn = model_smoteenn.predict(X_test)
        smoteenn_metrics = {
            "Accuracy": accuracy_score(y_test, y_pred_smoteenn),
            "Precision": precision_score(y_test, y_pred_smoteenn, zero_division=0),
            "Recall": recall_score(y_test, y_pred_smoteenn, zero_division=0),
            "F1-score": f1_score(y_test, y_pred_smoteenn, zero_division=0)
        }
        st.write("## Wyniki z SMOTEENN")
        st.write(smoteenn_metrics)

else:
    st.write("Proszę wgrać plik z danymi, aby kontynuować.")

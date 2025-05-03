import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import datetime
import matplotlib.pyplot as plt

def read_xls(file):
    df = pd.read_excel(file, engine='openpyxl')
    return df

def prepare_data(df):
    df['data'] = pd.to_datetime(df['data'])
    df['data_ordinal'] = df['data'].map(datetime.datetime.toordinal)
    X = df[['data_ordinal']]
    y = df['total_active_energy']
    return X, y

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict(model, X):
    return model.predict(X)

st.title("Total Active Energy Predictor")

uploaded_file = st.file_uploader("Upload an XLS or XLSX file", type=["xls", "xlsx"])

if uploaded_file is not None:
    df = read_xls(uploaded_file)
    
    if 'data' not in df.columns or 'total_active_energy' not in df.columns:
        st.error('Uploaded file should contain columns named "data" and "total_active_energy".')
    else:
        st.write("Data Preview:")
        st.write(df.head())

        X, y = prepare_data(df)
        model = train_model(X, y)

        st.write("Model trained. Provide a date range to predict the total active energy.")

        start_date = st.date_input("Select a start date")
        days_to_predict = st.slider("Number of days to predict (1-7)", 1, 7, 1)

        start_date_ordinal = datetime.datetime.toordinal(start_date)
        date_range = np.array(range(start_date_ordinal, start_date_ordinal + days_to_predict)).reshape(-1, 1)

        predictions = predict(model, date_range)

        # Display predictions
        prediction_dates = [datetime.datetime.fromordinal(date) for date in date_range.flatten()]
        pred_df = pd.DataFrame({
            'Date': prediction_dates,
            'Predicted Total Active Energy': predictions
        })
        st.write(pred_df)

        # Quality Metrics
        y_pred = predict(model, X)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        st.write(f"Mean Squared Error of trained model: {mse}")
        st.write(f"Mean Absolute Error of trained model: {mae}")
        st.write(f"R2 Score of trained model: {r2}")

        # Plotting
        fig, ax = plt.subplots()
        
        # Plot original data
        original_dates = [datetime.datetime.fromordinal(date) for date in X['data_ordinal']]
        ax.plot(original_dates, y, label="Original Data", marker='o')
        
        # Plot predicted data
        ax.plot(prediction_dates, predictions, label="Predicted Data", marker='x')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Total Active Energy')
        ax.legend()
        
        st.pyplot(fig)

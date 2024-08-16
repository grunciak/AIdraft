import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load the data
uploaded_file = st.file_uploader("Upload your Excel file with measurements", type=["xlsx"])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # Convert the date column to datetime format if not already
    df['Data'] = pd.to_datetime(df['Data'])

    st.write("### Data Preview")
    st.write(df.head())

    # Select column for forecasting
    column_to_forecast = st.selectbox("Select the column to forecast", df.columns[1:])

    # Show the selected column data
    st.write(f"### Selected Column: {column_to_forecast}")
    st.line_chart(df.set_index('Data')[column_to_forecast])

    # Prepare the data for modeling
    df['Timestamp'] = df['Data'].map(pd.Timestamp.timestamp)
    X = df[['Timestamp']]
    y = df[column_to_forecast]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate quality metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write("### Model Quality Metrics")
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"Mean Absolute Error: {mae}")
    st.write(f"R-squared: {r2}")

    # Predict future value
    forecast_date = st.date_input("Select a date for prediction")
    forecast_timestamp = pd.Timestamp(forecast_date).timestamp()
    forecast_value = model.predict(np.array([[forecast_timestamp]]))

    st.write(f"### Forecast for {forecast_date}: {forecast_value[0]}")

    # Plot actual and predicted values
    fig, ax = plt.subplots()
    ax.plot(df['Data'], df[column_to_forecast], label="Actual Data")
    ax.scatter(pd.to_datetime(X_test['Timestamp'], unit='s'), y_pred, color='red', label="Predicted Data")
    ax.axvline(pd.to_datetime(forecast_timestamp, unit='s'), color='green', linestyle='--', label="Forecast Point")
    ax.legend()
    st.pyplot(fig)

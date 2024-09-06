
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import datetime
import os

# Function to load and preprocess data from the local file
def load_data_from_file():
    # Path to the file in the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'heat_pump_data.xlsx')
    
    if os.path.exists(file_path):
        data = pd.read_excel(file_path, parse_dates={'data_czas': ['Data', 'Czas']}, decimal=',')
        data['timestamp'] = data['data_czas'].apply(lambda x: datetime.datetime.timestamp(x))
        return data
    else:
        st.error(f"File {file_path} not found.")
        return None

# Function to visualize the selected column over time
def plot_data(df, selected_column):
    fig = px.line(df, x='data_czas', y=selected_column, title=f'{selected_column} over Time')
    st.plotly_chart(fig, use_container_width=True)

# Function to perform regression using the Random Forest model
def perform_regression(df, selected_column):
    X = df[['timestamp']]
    y = df[selected_column]
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    return model, mse, r2, mae, y_pred

# Function to predict a future value based on the model, a specific date, and a specific time
def predict_value(model, date, time):
    combined_datetime = datetime.datetime.combine(date, time)
    timestamp = datetime.datetime.timestamp(combined_datetime)
    predicted_value = model.predict([[timestamp]])
    return predicted_value[0]

# Streamlit app layout and interaction
st.title('Legacy Heat Pump Data Analysis')

# Load data from the local file
data = load_data_from_file()

if data is not None:
    if st.button('Show Data'):
        st.write(data.head())
    
    selected_column = st.selectbox('Select a column to predict', data.columns[2:])
    
    if st.button('Predict'):
        plot_data(data, selected_column)
        
        # Train the model and store it in the session state
        model, mse, r2, mae, y_pred = perform_regression(data, selected_column)
        st.session_state['model'] = model
        
        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"R² Score: {r2}")
        st.write(f"Mean Absolute Error (MAE): {mae}")
        
        data['Prediction'] = y_pred
        fig2 = px.line(data, x='data_czas', y=[selected_column, 'Prediction'], title='Actual vs Predicted Data')
        st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader('Predict Future Value')

    date_input = st.date_input("Select a date for prediction")
    time_input = st.time_input("Select a time for prediction")

    if st.button('Predict Future Value'):
        # Ensure the model is available for predictions
        if 'model' in st.session_state:
            model = st.session_state['model']
            future_prediction = predict_value(model, date_input, time_input)
            st.write(f"Predicted value for {date_input} at {time_input}: {future_prediction}")
        else:
            st.write("Please train the model first by clicking 'Predict' for the initial data.")

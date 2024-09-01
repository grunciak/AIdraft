# Import necessary libraries

import streamlit as st  # Streamlit is used to create a web-based user interface.
import pandas as pd  # Pandas is used for data manipulation and analysis.
from sklearn.ensemble import RandomForestRegressor  # Random Forest model for regression.
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # Metrics to evaluate the model.
import plotly.express as px  # Plotly Express for interactive data visualization.
import datetime  # Datetime is used to handle date and time data.

# Function to load and preprocess data from an uploaded Excel file
def load_data(uploaded_file):
    # Read the Excel file into a Pandas DataFrame.
    # 'parse_dates' combines the 'Data' and 'Czas' columns into a single datetime column named 'data_czas'.
    # 'decimal' parameter handles the decimal point as a comma, which is common in some regions.
    data = pd.read_excel(uploaded_file, parse_dates={'data_czas': ['Data', 'Czas']}, decimal=',')
    
    # Convert the 'data_czas' datetime column into a Unix timestamp.
    # The timestamp is a numeric representation of time, which is easier for the model to process.
    data['timestamp'] = data['data_czas'].apply(lambda x: datetime.datetime.timestamp(x))
    
    # Return the processed DataFrame with the new 'timestamp' column.
    return data

# Function to visualize the selected column over time
def plot_data(df, selected_column):
    # Create a line plot using Plotly Express.
    # 'x' is the time ('data_czas') and 'y' is the selected data column.
    # The title of the plot dynamically reflects the selected column.
    fig = px.line(df, x='data_czas', y=selected_column, title=f'{selected_column} over Time')
    
    # Display the plot within the Streamlit app.
    st.plotly_chart(fig, use_container_width=True)

# Function to perform regression using the Random Forest model
def perform_regression(df, selected_column):
    # Define the independent variable (X) as the 'timestamp' and the dependent variable (y) as the selected column.
    X = df[['timestamp']]
    y = df[selected_column]
    
    # Initialize the Random Forest model with 100 decision trees (estimators).
    model = RandomForestRegressor(n_estimators=100)
    
    # Train (fit) the model on the provided data.
    model.fit(X, y)
    
    # Use the trained model to predict values for the data.
    y_pred = model.predict(X)
    
    # Calculate the Mean Squared Error (MSE) - lower is better.
    mse = mean_squared_error(y, y_pred)
    
    # Calculate the R² score - a value close to 1 indicates a good fit.
    r2 = r2_score(y, y_pred)
    
    # Calculate the Mean Absolute Error (MAE) - provides an average error in the same units as the data.
    mae = mean_absolute_error(y, y_pred)
    
    # Return the trained model and the calculated metrics.
    return model, mse, r2, mae, y_pred

# Function to predict a future value based on the model, a specific date, and a specific time
def predict_value(model, date, time):
    # Combine the provided date and time into a single datetime object.
    combined_datetime = datetime.datetime.combine(date, time)
    
    # Convert the combined datetime into a Unix timestamp.
    timestamp = datetime.datetime.timestamp(combined_datetime)
    
    # Use the trained model to predict the value for the given timestamp.
    predicted_value = model.predict([[timestamp]])
    
    # Return the predicted value (the first and only item in the result).
    return predicted_value[0]

# Streamlit app layout and interaction

# Set the title of the Streamlit app.
st.title('Legacy Heat Pump Data Analysis')

# Create a file uploader widget to allow users to upload an Excel file.
uploaded_file = st.file_uploader("Select an Excel file", type=["xlsx"])

# Check if a file has been uploaded.
if uploaded_file is not None:
    # Load and preprocess the data using the 'load_data' function.
    data = load_data(uploaded_file)
    
    # Button to show the first few rows of the data.
    if st.button('Show Data'):
        # Display the first few rows of the DataFrame.
        st.write(data.head())
    
    # Dropdown menu to select which column to analyze and predict.
    selected_column = st.selectbox('Select a column to predict', data.columns[2:])
    
    # Button to trigger the prediction process.
    if st.button('Predict'):
        # Visualize the selected column over time.
        plot_data(data, selected_column)
        
        # Perform regression analysis on the selected column.
        model, mse, r2, mae, y_pred = perform_regression(data, selected_column)
        
        # Display the calculated performance metrics.
        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"R² Score: {r2}")
        st.write(f"Mean Absolute Error (MAE): {mae}")
        
        # Add the model's predictions to the DataFrame.
        data['Prediction'] = y_pred
        
        # Create a plot comparing the actual data with the model's predictions.
        fig2 = px.line(data, x='data_czas', y=[selected_column, 'Prediction'], title='Actual vs Predicted Data')
        
        # Display the comparison plot in the Streamlit app.
        st.plotly_chart(fig2, use_container_width=True)
    
    # Section for predicting future values
    st.subheader('Predict Future Value')

    # Input for selecting a future date
    date_input = st.date_input("Select a date for prediction")

    # Input for selecting a specific time on the selected date
    time_input = st.time_input("Select a time for prediction")

    # Button to predict the value for the specific date and time
    if st.button('Predict Future Value'):
        # Call the predict_value function with the selected date and time
        future_prediction = predict_value(model, date_input, time_input)
        
        # Display the predicted value
        st.write(f"Predicted value for {date_input} at {time_input}: {future_prediction}")

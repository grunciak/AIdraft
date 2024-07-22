import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import datetime

def load_data(uploaded_file):
    data = pd.read_excel(uploaded_file, parse_dates={'data_czas': ['Data', 'Czas']}, decimal=',')
    data['timestamp'] = data['data_czas'].apply(lambda x: datetime.datetime.timestamp(x))
    return data

def plot_data(df, selected_column):
    fig = px.line(df, x='data_czas', y=selected_column, title=f'{selected_column} w czasie')
    st.plotly_chart(fig, use_container_width=True)

def perform_regression(df, selected_column):
    X = df[['timestamp']]
    y = df[selected_column]
    model = RandomForestRegressor(n_estimators=100)  # Używamy Random Forest z 100 drzewami
    model.fit(X, y)
    y_pred = model.predict(X)
    
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    return model, mse, r2, mae, y_pred

def predict_value(model, df, date):
    timestamp = datetime.datetime.timestamp(date)
    predicted_value = model.predict([[timestamp]])
    return predicted_value[0]

st.title('Analiza danych')

uploaded_file = st.file_uploader("Wybierz plik xlsx", type=["xlsx"])
if uploaded_file is not None:
    data = load_data(uploaded_file)
    if st.button('Pokaż dane'):
        st.write(data.head())
    
    selected_column = st.selectbox('Wybierz kolumnę do przewidywania', data.columns[2:])
    
    if st.button('Przewiduj'):
        plot_data(data, selected_column)
    
    model, mse, r2, mae, y_pred = perform_regression(data, selected_column)
    st.write(f"Błąd średniokwadratowy (MSE): {mse}")
    st.write(f"Współczynnik determinacji (R^2): {r2}")
    st.write(f"Mean Absolute Error (MAE): {mae}")  
    
    data['Predykcja'] = y_pred
    fig2 = px.line(data, x='data_czas', y=[selected_column, 'Predykcja'], title='Rzeczywiste vs Prognozowane dane')
    st.plotly_chart(fig2, use_container_width=True)

    selected_date = st.date_input("Wybierz datę do prognozowania", datetime.date.today())
    if st.button('Prognozuj wartość'):
        predicted_value = predict_value(model, data, datetime.datetime.combine(selected_date, datetime.datetime.min.time()))
        st.write(f"Prognozowana wartość dla {selected_date}: {predicted_value}")
        
        fig3 = px.line(data, x='data_czas', y=selected_column, title=f'Prognoza dla {selected_column}')
        fig3.add_scatter(x=[selected_date], y=[predicted_value], mode='markers+text', text=["Prognoza"], name="Prognoza")
        st.plotly_chart(fig3, use_container_width=True)

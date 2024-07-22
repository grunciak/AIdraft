
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Tytuł aplikacji
st.title('Przewidywanie zdarzeń na podstawie daty i pomiarów')

# Wczytywanie plików Excel
st.header('Wczytaj pliki z alarmami i pomiarami')

uploaded_file_alarms = st.file_uploader("Wybierz plik z alarmami", type=["xlsx"])
uploaded_file_measurements = st.file_uploader("Wybierz plik z pomiarami", type=["xlsx"])

if uploaded_file_alarms and uploaded_file_measurements:
    # Wczytywanie danych z plików
    df_alarms = pd.read_excel(uploaded_file_alarms)
    df_measurements = pd.read_excel(uploaded_file_measurements)
    
    # Wyświetlanie danych
    st.subheader("Dane z alarmami:")
    st.write(df_alarms)
    
    st.subheader("Dane z pomiarami:")
    st.write(df_measurements)
    
    # Przyjmujemy, że obie tabele mają wspólną kolumnę z datą do połączenia
    if 'date' in df_alarms.columns and 'date' in df_measurements.columns:
        df_alarms['date'] = pd.to_datetime(df_alarms['date'])
        df_measurements['date'] = pd.to_datetime(df_measurements['date'])
        
        # Łączenie danych na podstawie kolumny z datą
        df = pd.merge(df_measurements, df_alarms, on='date', how='inner')
        
        # Wyświetlanie połączonych danych
        st.subheader("Połączone dane:")
        st.write(df)
        
        # Przyjmujemy, że kolumna 4 zawiera informacje o zdarzeniach
        if 4 in df.columns:
            df['event'] = df[4]
            
            # Wybór kolumn do predykcji
            features = st.multiselect("Wybierz kolumny do użycia jako cechy (features)", options=df.columns)
            
            if features:
                X = df[features]
                y = df['event']
                
                # Podział danych na zbiór treningowy i testowy
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Trenowanie modelu
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                
                # Przewidywanie
                y_pred = model.predict(X_test)
                
                # Wyświetlanie wyników
                st.subheader("Wyniki predykcji:")
                st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
                st.write("Classification Report:")
                st.text(classification_report(y_test, y_pred))
                
                # Możliwość przewidywania dla nowych danych
                st.header('Przewidywanie dla nowych danych')
                input_data = {feature: st.number_input(f"Wprowadź wartość dla {feature}") for feature in features}
                
                if st.button("Przewiduj"):
                    input_df = pd.DataFrame([input_data])
                    prediction = model.predict(input_df)
                    st.write(f"Przewidywane zdarzenie: {prediction[0]}")
        else:
            st.error("Dane alarmowe nie zawierają kolumny 4.")
    else:
        st.error("Oba pliki muszą zawierać kolumnę 'date' do połączenia danych.")
else:
    st.info("Proszę wczytać oba pliki Excel.")

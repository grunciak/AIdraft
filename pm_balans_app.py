import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import datetime
import pickle
import plotly.express as px
import shap  # <-- NOWE: SHAP
import matplotlib.pyplot as plt

# Cache danych – szybsze działanie
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "2810Merged_PM_Data2.xlsx")
    try:
        data = pd.read_excel(file_path)
        data["date"] = pd.to_datetime(data["date"])
        return data
    except FileNotFoundError:
        st.error("Brak pliku 2810Merged_PM_Data2.xlsx!")
        st.stop()

def app():
    st.title('Predykcja Alarmów + SHAP')

    data = load_data()

    # --- Wybór alarmu (tylko te, które coś mają) ---
    exclude = ["CT-RS485 Temperatura", "KTW-2 Temperatura", "KTW-2 Wilgotność", "hour", "day_of_week", "date"]
    alarm_columns = [c for c in data.columns if c not in exclude and data[c].sum() > 0]

    selected_alarm = st.selectbox("Wybierz alarm", alarm_columns)

    # --- Przygotowanie danych ---
    data[selected_alarm] = data[selected_alarm].fillna(0)
    data["hour"] = data["date"].dt.hour
    data["day_of_week"] = data["date"].dt.dayofweek

    features = [c for c in data.columns if c not in ["date", selected_alarm]]
    X = data[features]
    y = (data[selected_alarm] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # --- Trening modeli ---
    models = {}

    # Najlepszy balans: scale_pos_weight (działa zawsze)
    scale = (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1
    model = XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=scale)
    model.fit(X_train, y_train)
    models["XGBoost (scale_pos_weight)"] = model

    # Undersampling – zawsze działa
    rus = RandomUnderSampler(random_state=42)
    X_u, y_u = rus.fit_resample(X_train, y_train)
    model_u = XGBClassifier(random_state=42, eval_metric='logloss')
    model_u.fit(X_u, y_u)
    models["Undersampling"] = model_u

    # SMOTE – tylko jeśli jest dość próbek
    if y_train.sum() > 10:
        try:
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_s, y_s = smote.fit_resample(X_train, y_train)
            model_s = XGBClassifier(random_state=42, eval_metric='logloss')
            model_s.fit(X_s, y_s)
            models["SMOTE"] = model_s
        except:
            pass

    # --- Średnie prawdopodobieństwo ---
    st.subheader("Średnie przewidywane prawdopodobieństwo alarmu")
    probs = []
    for name, m in models.items():
        prob = m.predict_proba(X_test)[:, 1].mean()
        probs.append({"Model": name, "Średnie prawdopodobieństwo": f"{prob:.2%}"})
    st.table(probs)

    # --- Korelacje między alarmami ---
    st.subheader("Korelacje między alarmami")
    corr = data[alarm_columns].corr()
    st.dataframe(corr.style.background_gradient(cmap='coolwarm').format("{:.2f}"))

    # --- Predykcja ---
    st.subheader("Predykcja dla wybranej daty i godziny")
    col1, col2 = st.columns(2)
    with col1:
        pred_date = st.date_input("Data", value=datetime.date.today() + datetime.timedelta(days=1))
    with col2:
        pred_time = st.time_input("Godzina", value=datetime.time(12, 0))

    model_name = st.selectbox("Model do predykcji i SHAP", list(models.keys()))
    model = models[model_name]

    if st.button("Przewiduj + wyjaśnij (SHAP)", type="primary"):
        # Przygotowanie wiersza
        last_row = data.iloc[-1].copy()
        dt = datetime.datetime.combine(pred_date, pred_time)
        last_row["hour"] = dt.hour
        last_row["day_of_week"] = dt.weekday()

        # Lepsza aproksymacja temperatur (średnia z podobnych dni/godzin)
        similar = data[(data['day_of_week'] == dt.weekday()) & (data['hour'] == dt.hour)]
        if len(similar) > 5:
            last_row[['CT-RS485 Temperatura', 'KTW-2 Temperatura', 'KTW-2 Wilgotność']] = similar[
                ['CT-RS485 Temperatura', 'KTW-2 Temperatura', 'KTW-2 Wilgotność']].mean()

        X_pred = last_row[features].values.reshape(1, -1)

        pred = model.predict(X_pred)[0]
        prob = model.predict_proba(X_pred)[0][1]

        st.write(f"### {pred_date} {pred_time.strftime('%H:%M')}")
        if pred == 1:
            st.error("ALARM PRZEWIDYWANY")
            st.warning(f"Prawdopodobieństwo: **{prob:.1%}**")
        else:
            st.success("Brak alarmu")
            st.info(f"Prawdopodobieństwo: **{prob:.1%}**")

        # Rzeczywistość (jeśli data istnieje)
        real = data[data["date"].dt.date == pred_date]
        if len(real) > 0:
            actual = real[selected_alarm].iloc[0]
            st.info(f"Rzeczywistość: {'ALARM' if actual > 0 else 'Brak alarmu'}")

        # =====================================
        #          SHAP – wyjaśnienia
        # =====================================
        st.subheader("Wyjaśnienie predykcji (SHAP)")

        # Explainer – TreeExplainer jest najszybszy dla XGBoost
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_pred)

        # 1. Force plot (interaktywny)
        st.components.v1.html(
            shap.getjs(),
            height=0, width=0
        )
        force_plot = shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            X_pred[0],
            feature_names=features,
            matplotlib=False,
            show=False
        )
        st.components.v1.html(shap.html(force_plot), height=200, scrolling=True)

        # 2. Waterfall (najlepszy do zrozumienia jednej predykcji)
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_pred[0],
                feature_names=features
            ),
            show=False
        )
        st.pyplot(fig)

        # 3. Summary dla całego modelu (top 10 cech)
        st.write("Top 10 najważniejszych cech dla tego modelu (cały zbiór testowy):")
        shap_test = explainer.shap_values(X_test.sample(frac=0.3, random_state=42))
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_test, X_test.sample(frac=0.3, random_state=42), feature_names=features, show=False, max_display=10)
        st.pyplot(fig2)

if __name__ == "__main__":
    app()
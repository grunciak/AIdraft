# hub.py – wersja BEZ logowania
import streamlit as st
import pm_balans_app  # Twoja apka z funkcją app()


def main():
    st.set_page_config(page_title="Portal skryptów PM", layout="wide")

    st.title("Portal skryptów predykcyjnych")
    st.sidebar.success("Jesteś zalogowany automatycznie")

    # Lista dostępnych aplikacji
    apps = {
        "Predykcja alarmów": pm_balans_app,
        # Tu możesz dodać kolejne w przyszłości:
        # "Zużycie wody": woda_app,
        # "Pompa ciepła": pompa_app,
    }

    choice = st.sidebar.radio("Wybierz aplikację:", list(apps.keys()))

    # Uruchamiamy wybraną apkę
    selected_app = apps[choice]
    selected_app.app()


if __name__ == "__main__":
    main()
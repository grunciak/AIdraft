import streamlit as st
import streamlit_authenticator as stauth

import pm_balans_app  # nasza apka z funkcją app()


def main():
    st.set_page_config(page_title="Portal skryptów PM", layout="wide")

    # --- CREDENTIALS Z JUŻ ZAHASHOWANYM HASŁEM ---
    credentials = {
        "usernames": {
            "admin": {
                "email": "admin@example.com",
                "first_name": "Admin",
                "last_name": "User",
                # TU WKLEJ HASH Z generate_hash.py:
                "password": "$2b$12$Lwk/FKLwoKKUjaGL7F15aek5h..e2WEe4EMxGq9B3ul8aheSTpR4a",
                # failed_login_attempts i logged_in dodadzą się same
            }
        }
    }

    # auto_hash=False → NIE hashujemy ponownie, hasła są już zhashowane
    authenticator = stauth.Authenticate(
        credentials=credentials,
        cookie_name="epm_pm_cookie",
        cookie_key="epm_pm_signature_key",  # przy wdrożeniu zmień na losowy długi string
        cookie_expiry_days=1,
        auto_hash=False,
    )

    # --- LOGOWANIE ---
    try:
        authenticator.login(location="main")
    except Exception as e:
        st.error(e)
        return

    auth_status = st.session_state.get("authentication_status", None)
    name = st.session_state.get("name", None)
    username = st.session_state.get("username", None)

    if auth_status is False:
        st.error("Błędny login lub hasło.")
        return
    elif auth_status is None:
        st.warning("Wpisz login i hasło.")
        return

    # --- ZALOGOWANY ---
    authenticator.logout("Wyloguj", "sidebar")

    if name:
        st.sidebar.write(f"Zalogowany jako: {name}")
    elif username:
        st.sidebar.write(f"Zalogowany jako: {username}")

    st.title("Portal skryptów predykcyjnych")
    st.write("Wybierz skrypt z menu po lewej stronie.")

    apps = {
        "Predykcja alarmów": pm_balans_app,
    }

    choice = st.sidebar.radio("Wybierz aplikację:", list(apps.keys()))
    selected_app = apps[choice]
    selected_app.app()


if __name__ == "__main__":
    main()

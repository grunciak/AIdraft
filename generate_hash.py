from streamlit_authenticator.utilities.hasher import Hasher

if __name__ == "__main__":
    plain = "predykcja123"  
    hashed = Hasher.hash(plain)
    print("Hashed password:")
    print(hashed)

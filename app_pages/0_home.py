import streamlit as st


def show():
    for key, _ in st.session_state.items():
        del st.session_state[key]

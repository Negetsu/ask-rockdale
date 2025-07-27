import streamlit as st
st.write("Hello Worlde")
user_text = st.text_input("Favourite Word?")
st.write(f"The User's favorite word is {user_text}")
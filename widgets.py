import streamlit as st
import pandas as pd

st.title("Streamlit Text Input")

# Display a simple text input
name = st.text_input("Enter your name: ")
if name:
    st.write(f"Hello, {name}!")

# Display a slider for age selection
age = st.slider("Select your age:", 0, 100, 18)

st.write(f"You are {age} years old.")


# Display a selectbox
options = ["Python", "Java", "C++", "JavaScript"]
choice = st.selectbox("Choose a programming language:", options)
st.write(f"You selected: {choice}")

# Create a DataFrame and display it
df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["New York", "Los Angeles", "Chicago"]
})
df.to_csv("data.csv")
df.to_excel("sample.xlsx")
st.write(df)

# Upload a CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)

# Task: Add a file uploader for Excel files
uploaded_excel = st.file_uploader("Upload an Excel file", type=["xlsx"])
if uploaded_excel is not None:
    df_excel = pd.read_excel(uploaded_excel)
    st.write(df_excel)
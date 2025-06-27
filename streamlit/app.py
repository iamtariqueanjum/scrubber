import streamlit as st
import pandas as pd
import numpy as np

# Title of the Streamlit app
st.title("Welcome to the Streamlit App")

# Display a simple text
st.write("This is a simple text")

# Create a DataFrame
df = pd.DataFrame({
    "Column 1": [1, 2, 3, 4, 5],
    "Column 2": [10, 20, 30, 40, 50],
    "Column 3": [100, 200, 300, 400, 500]
})

# Display the DataFrame
st.write(df)

# Create a line chart

chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=["a", "b", "c"]
    )

# Display the line chart
st.line_chart(chart_data)
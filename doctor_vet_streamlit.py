import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Title of the app
st.title("Doctor and Veterinary Classification App")

# Display text
st.write("This notebook is for building a model which will correctly classify a number of given reddit users as practicing doctors, practicng veterinary or others based on each user's comments")

# Display text
st.write("The dataset for this task would be sourced from a database whose link is given as")

st.write("[postgresql://niphemi.oyewole:W7bHIgaN1ejh@ep-delicate-river-a5cq94ee-pooler.us-east-2.aws.neon.tech/Vetassist?statusColor=F8F8F8&env=&name=redditors%20db&tLSMode=0&usePrivateKey=false&safeModeLevel=0&advancedSafeModeLevel=0&driverVersion=0&lazyload=false](postgresql://niphemi.oyewole:W7bHIgaN1ejh@ep-delicate-river-a5cq94ee-pooler.us-east-2.aws.neon.tech/Vetassist?statusColor=F8F8F8&env=&name=redditors%20db&tLSMode=0&usePrivateKey=false&safeModeLevel=0&advancedSafeModeLevel=0&driverVersion=0&lazyload=false)")

# Header
st.header('Introduction to Streamlit')

# Display text
st.write('Streamlit is an open-source app framework for Machine Learning and Data Science projects.')

# Create a dataframe
df = pd.DataFrame({
    'Column 1': np.random.randn(10),
    'Column 2': np.random.randn(10)
})

# Display the dataframe
st.write('Here is a random dataframe:')
st.write(df)

# Plotting
st.write('Here is a simple line plot:')
fig, ax = plt.subplots()
ax.plot(df['Column 1'], label='Column 1')
ax.plot(df['Column 2'], label='Column 2')
ax.legend()
st.pyplot(fig)

# Add a slider
slider_value = st.slider('Select a value', 0, 100, 50)
st.write(f'Slider value is: {slider_value}')

# Add a text input
text_input = st.text_input('Enter some text')
st.write(f'You entered: {text_input}')

# Add a button
if st.button('Click me'):
    st.write('Button clicked!')

# Add a selectbox
option = st.selectbox('Select an option', ['Option 1', 'Option 2', 'Option 3'])
st.write(f'You selected: {option}')

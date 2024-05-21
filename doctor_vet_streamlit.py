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

# import modules
import re             # for regrex operations
import string         # for removing punctuations
import random         # for generating random numbers
import statistics     # for statistical functions
import numpy as np    # for mathematical calculations
import pandas as pd   # for working with structured data (dataframes)
import matplotlib.pyplot as plt                   # for making plots
from xgboost import XGBClassifier                 # XGBoost model
from nltk.corpus import stopwords                 # for getting stopwords
from sqlalchemy import create_engine              # for connecting to database
from nltk.tokenize import word_tokenize           # for tokenizing words
from sklearn.metrics import accuracy_score        # for getting prediction accuracy
from sklearn.naive_bayes import MultinomialNB     # Multinimial Naive Bayes model
from sklearn.preprocessing import LabelEncoder    # for encoding target class
from sklearn.ensemble import AdaBoostClassifier   # AdaBoost model
from sklearn.tree import DecisionTreeClassifier   # Decision Tree model
from sklearn.ensemble import StackingClassifier   # Stacking Ensemble model
from sklearn.metrics import classification_report     # for generating classification report
from sklearn.neighbors import KNeighborsClassifier    # k Nearest Neighbour model
from sklearn.model_selection import train_test_split  # for splitting into trainning and test set
from sklearn.feature_extraction.text import TfidfVectorizer  # for vectorizing words

# define the connection link to database
conn_str = "postgresql://niphemi.oyewole:endpoint=ep-delicate-river-a5cq94ee-pooler;W7bHIgaN1ejh@ep-delicate-river-a5cq94ee-pooler.us-east-2.aws.neon.tech/Vetassist?sslmode=allow"

# create connection to the databse
engine =  create_engine(conn_str)

st.write("First, lets take a look at the tables in the database")



# Header
st.header('Module Importations and Data Retrieval')

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

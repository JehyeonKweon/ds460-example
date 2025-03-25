import streamlit as st
import pandas as pd
import joblib


st.title("Predict Gender with Height and Shoe Size")

@st.cache_resource

def load_medel():
    return joblib.load('./naive_bayes_gender_model_US.pkl')

model = load_medel()

st.subheader('Enter Height and Shoe size')

col1, col2 = st.columns(2)

with col1:
    feet = st.number_input('Height (ft)', min_value=1, max_value=10, value=5, step=1)

with col2:
    inches = st.number_input('Height (ft)', min_value=0, max_value=11, value=6, step=1)

height = feet + (inches / 12)

normalized_shoe_size = st.number_input('Shoe Size (US)', min_value=1.0, max_value=24.0, value=10.0, step=0.5)

if st.button('OK'):

    data = pd.DataFrame({'height': [height], 'normalized_shoe_size': [normalized_shoe_size]})

    predictions = model.predict(data)[0]

    pred_dict = {1:'Male', 0:'Female'}

    st.write(f'I think you are {pred_dict[predictions]}.')
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Heart sick Prediction App

This app predicts the **some heart sick type!** 

""")

st.sidebar.header('User Input Features')



# Collects user input features into dataframe

def user_input_features():
    Weight= st.sidebar.slider('Weight', 0.0, 1650.0, 1000.0)
    Width = st.sidebar.slider('Width', 1.0, 8.15, 6.0)
    Height = st.sidebar.slider('Height', 1.72, 18.96, 10.00)
    Length1 = st.sidebar.slider('Length1', 7.5, 59.0, 30.0)
    Length2 = st.sidebar.slider('Length2', 8.4, 63.4, 40.0)
    Length3 = st.sidebar.slider('Length3', 8.8, 68.0, 40.0)
    data = {'Weight': Weight,
                'Width': Width,
                'Height': Height,
                'Length1': Length1,
                'Length2': Length2,
                'Length3': Length3,}
    features = pd.DataFrame(data, index=[0])
    return features
df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

hert = pd.read_csv('fish_data.csv')
X = hert.drop('Species', axis= 1)
Y = hert['Species']

clf = RandomForestClassifier()
clf.fit(X, Y)

heart_cardio = np.array([0, 1, 2, 3, 4, 5, 6])

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(heart_cardio)

st.subheader('Prediction')
st.write(heart_cardio[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
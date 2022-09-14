import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Customer Scoring Prediction App

This app predicts the **Customer Scoring** !
""")

st.sidebar.header('Customer')

wine = pd.read_csv("wine_new.csv")
print(wine.describe(include='all'))

def user_input_features():
    Alcohol = st.sidebar.slider('Alcohol', 11.0, 15.0, 12.0)
    Ash = st.sidebar.slider('Ash', 0.1, 4.0, 2.0)
    Acl = st.sidebar.slider('ACL', 10.0, 31.0, 10.0)
    Flavanoids = st.sidebar.slider('Flavanoids', 0.1, 6.0, 1.0)
    Nonflavanoidphenols = st.sidebar.slider('Nonflavanoidphenols', 0.0, 1.0, 0.3)
    Colorint = st.sidebar.slider('Nonflavanoidphenols', 1.0, 15.0, 2.0)  
    data = {'Alcohol': Alcohol,
            'Ash': Ash,
            'Acl': Acl,
            'Flavanoids': Flavanoids,
            'Nonflavanoidphenols': Nonflavanoidphenols,
            'Colorint': Colorint}
    features = pd.DataFrame(data,index=[0])
    return features
df = user_input_features()


#print(wine.describe(include='all'))

X = wine.drop('Wine', axis=1)
#print(X.describe(include='all'))
Y = wine['Wine']

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.subheader('User Input parameters')
st.write(df)

st.subheader('Class labels and their corresponding index number')
st.write(['1' '2' '3'])

st.subheader('Prediction')
st.write([prediction])

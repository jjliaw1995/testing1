##writefile myfirstapp.py
import streamlit as st
import numpy as np
import pandas as pd

st.header("Presenting the data of iris")

st.write(pd.DataFrame({
'0': ["Iris-setosa"],
'1': ["Iris-versicolor"],
'2': ["Iris-virginica"]
}))

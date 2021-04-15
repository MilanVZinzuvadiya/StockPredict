import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from stock_predict import *

@st.cache
def stock_predict(ticker,test_pr):
    return run(ticker,test_pr)

st.write("""
# Stock Prediction App
This app predicts the **Stock Prices** !
""")

st.sidebar.header('Prediction Setting')

def user_input_features():
    test_pr = st.sidebar.slider('Test data percentage %', 1, 20, 10)
    stock = st.sidebar.text_input("Enter ticker of stock", "NFLX")
    data = {'test_pr': test_pr,
            'stock': stock}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
st.subheader('User Input parameters')
st.table(df)

predict_data,train = stock_predict(df['stock'][0],df['test_pr'][0])
predict_data = predict_data[["Adj Close","AutoRegression"]]

st.line_chart(predict_data)
st.line_chart(train['Adj Close'])
import streamlit as st
from stock_predict import *

@st.cache
def stock_predict_auto(ticker,test_pr):
    return run(ticker,test_pr)

@st.cache
def stock_predict_arima(ticker,test_pr):
    return run(ticker,test_pr,'arima')

@st.cache
def stock_predict_ses(ticker,test_pr):
    return run(ticker,test_pr,'ses')

@st.cache
def stock_predict_es(ticker,test_pr):
    return run(ticker,test_pr,'es')

st.write("""
# Stock Prediction App
This app predicts the **Stock Prices** !
""")
st.subheader('by Milan Zinzuvadiya')
st.sidebar.header('Prediction Setting')

def user_input_features():
    usr_par = {}
    usr_par['test_pr'] = st.sidebar.slider('Test data percentage %', 1, 20, 10)
    usr_par['stock'] = st.sidebar.text_input("Enter ticker of stock", "NFLX")

    list_algo = ['AutoRegression','Moving Average','Simple Exponential Smoothing','Exponential Smoothing']
    selected_algo = []
    for i in list_algo:
        selected_algo.append(st.sidebar.checkbox(i,value=True))
    usr_par['selected_algo'] = selected_algo
    seasons = 0
    
    usr_par['seasons'] = seasons
    
    return usr_par

df = user_input_features()

compare_graph = pd.DataFrame()

# 0 index is for autoregression 
if df['selected_algo'][0]:
    predict_data,train = stock_predict_auto(df['stock'],df['test_pr'])
    compare_graph = predict_data

# 1 index is for sarimax
if df['selected_algo'][1]:
    predict_data, train = stock_predict_arima(df['stock'],df['test_pr'])
    compare_graph = pd.concat([compare_graph,predict_data],axis=1)

# 2 index is for Simple Exponential Smoothing
if df['selected_algo'][2]:
    predict_data, train = stock_predict_ses(df['stock'],df['test_pr'])
    compare_graph = pd.concat([compare_graph,predict_data],axis=1)

# 3 index is for Exponential Smoothing
if df['selected_algo'][3]:
    predict_data, train = stock_predict_es(df['stock'],df['test_pr'])
    compare_graph = pd.concat([compare_graph,predict_data],axis=1)

st.subheader('Prediction of ' + str(df['test_pr'])+' % data of '+df['stock'])
st.line_chart(compare_graph)
st.line_chart(train['Adj Close'])


#dtr = pd.DataFrame(train['Adj Close']).append(compare_graph,sort=False)


#dtf = predict_data.append(pd)
#fg = pd.concat([predict_data_auto,predict_data_arima],axis=1)


#st.write(dtf)
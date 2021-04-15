import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.statespace.sarimax import SARIMAX

predict_label = 'Adj Close'

def getData(ticker):
    return yf.download(ticker, period="max")



def train_test(test_pr,ticker):
    test_pr = test_pr/100.0
    data = getData(ticker)
    dataLength = len(data)
    split_point = int((1-test_pr)*dataLength)
    train = data[:split_point]
    test = data[split_point:]
    return train,test


#Forecast
def train_auto(train,test):
    y_hat = test.copy()
    forecast_name= 'AutoRegression'
    model = AutoReg(train['Adj Close'],lags=1)
    model_fit = model.fit()
    return model_fit,y_hat

def train_sarimax(n_seasons,train):
    y_hat = test.copy()
    model = SARIMAX(train[predict_label],order=(1,1,1),seasonal_order=(1,1,1,n_seasons))
    model_fit = model.fit()
    return model_fit,y_hat

def predict(model,train,test,y_hat):
    return np.array(model.predict(start=len(train),end=len(train)+len(test)-1))
#prediction

def run(ticker,test_pr,model_name='auto',n_seasons=4):
    train,test = train_test(test_pr,ticker)
    if model_name == 'auto':
        model,y_hat = train_auto(train,test)
        forecast_name = 'AutoRegression'
    else:
        model,y_hat = train_sarimax(n_seasons,train)
        forecast_name = 'SARIMAX'
    
    y_hat[forecast_name] = predict(model,train,test,y_hat)
    return y_hat,train
    


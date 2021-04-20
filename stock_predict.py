import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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
    model = AutoReg(train['Adj Close'],lags=1)
    model_fit = model.fit()
    return model_fit,y_hat

def train_arimax(train,test):
    y_hat = test.copy()
    model = ARIMA(train[predict_label],order=(5,1,1))
    model_fit = model.fit()
    return model_fit,y_hat

def train_ses(train,test):
    y_hat = test.copy()
    model = SimpleExpSmoothing(train[predict_label])
    model_fit = model.fit()
    return model_fit,y_hat

def train_es(train,test):
    y_hat = test.copy()
    model = ExponentialSmoothing(train[predict_label])
    model_fit = model.fit()
    return model_fit,y_hat

def predict(model,train,test,y_hat):
    return np.array(model.predict(start=len(train),end=len(train)+len(test)-1))
#prediction

def run(ticker,test_pr,model_name='auto'):
    train,test = train_test(test_pr,ticker)
    if model_name == 'auto':
        model,y_hat = train_auto(train,test)
        forecast_name = 'AutoRegression'
    elif model_name == 'ses':
        model,y_hat = train_ses(train,test)
        forecast_name = 'Simple-Exp-Smoothing'
    elif model_name == 'es':
        model,y_hat = train_es(train,test)
        forecast_name = 'Expo-Smoothing'
    else:
        model,y_hat = train_arimax(train,test)
        forecast_name = 'Moving Average'
    
    y_hat[forecast_name] = predict(model,train,test,y_hat)
    return y_hat[["Adj Close",forecast_name]],train
    


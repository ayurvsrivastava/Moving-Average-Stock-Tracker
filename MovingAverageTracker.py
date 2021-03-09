#Necessary Imports 
import os
import random
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from collections import deque

#Debugging for consistent results
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)

#Similiar Shuffle for arrays
def sameShuffle ( arrayA , arrayB ):
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)

def tickerData ( ticker, windowSize, predictionStep, test_size, columnNames):

    #Load Yahoo Finance Data
    dataframe = yf.Ticker( ticker ).history(period = "5y")

    #Store All Final Data
    results = {}
    
    #Create Usable Dates Column
    dataframe["date"] = dataframe.index

    #Copy Dataframe
    results['dataframe'] = dataframe.copy()

    #Scale Values
    scaleColumns = {}
    for col in columnNames:
        scaler = preprocessing.MinMaxScaler()
        dataframe[col] = scaler.fit_transform(np.expand_dims(dataframe[col].values, axis = 1))
        scaleColumns[col] = scaler
    results['scaleColumns'] = scaleColumns

    #Begin Future Calculations to predict sotck prices within data set
    dataframe['future'] = dataframe['Close'].shift(-predictionStep)

    #Get Final Values then frop NaN values
    tailValues = np.array(dataframe[columnNames].tail(predictionStep))
    dataframe.dropna(inplace=True)
    
    #Begin Data Filterting
    dataSequence = []
    data = deque( maxlen = windowSize )
    for entry, target in zip(dataframe[columnNames + ["date"]].values, dataframe['future'].values):
        data.append(entry)
        if len(data) == windowSize:
            dataSequence.append([np.array(data), target])

    #Predict stock prices for values not in dataset
    tailValues = list([s[:len(columnNames)] for s in data]) + list(tailValues)
    tailValues = np.array(tailValues).astype(np.float32)

    #Add prediction to results
    results['tailValues'] = tailValues

    #begin training datasets
    x, y = [], []
    for seq, target in dataSequence:
        x.append(seq)
        y.append(target)
    
    x = np.array(x)
    y = np.array(y)

    results["x_train"], results["x_test"], results["y_train"], results["y_test"] = train_test_split(x, y, test_size = test_size, shuffle = True)

    #store test dates
    dates = results["x_test"][:, -1, 1]
    results["test_dataframe"] = results["dataframe"].loc[dates]

    #remove same results
    results["test_dataframe"] = results["test_dataframe"][~results["test_dataframe"].index.duplicate(keep='first')]

    #Convert dada into float32 for processing
    results["x_train"] = results["x_train"][:, :, :len(columnNames)].astype(np.float32)
    results["x_test"] = results["x_test"][:, :, :len(columnNames)].astype(np.float32)
    
    return results

#Create Model
def stock_model( windowSteps, nFeatures, loss, units, cell, nLayers, dropout, optimizer, bidirectional):
    finalModel = Sequential()
    for i in range(nLayers):
        if i == 0:
                # first layer
            if bidirectional:
                finalModel.add(Bidirectional(cell(units, return_sequences=True), batch_input_shape=(None, windowSteps, nFeatures)))
            else:
                finalModel.add(cell(units, return_sequences=True, batch_input_shape=(None, windowSteps, nFeatures)))
        elif i == nLayers - 1:
            # last layer
            if bidirectional:
                finalModel.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                finalModel.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                finalModel.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                finalModel.add(cell(units, return_sequences=True))
        # add dropout after each layer
        finalModel.add(Dropout(dropout))
    finalModel.add(Dense(1, activation="linear"))
    finalModel.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return finalModel



ticker = "MSFT"
windowSize = 50
predictionStep = 15
test_size = 0.2
columnNames = ["Open", "High", "Low", "Close", "Volume"]
dateNow = time.strftime("%Y-%m-%d")
nLayers = 2
cell = LSTM
units = 256
dropout = .4
bidirectional = False
loss = "huber_loss"
optimizer = "adam"
batch_size = 64
epochs = 500

if not os.path.isdir("results"):
    os.mkdir("results")
if not os.path.isdir("logs"):
    os.mkdir("logs")
if not os.path.isdir("data"):
    os.mkdir("data")

tickerResults = tickerData( ticker, windowSize, predictionStep, test_size, columnNames)

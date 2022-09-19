from statistics import mean
from math import sqrt
import pandas as pd
import numpy as np

def mseFunction(test_data, pred_data):
    dif = test_data - pred_data
    dif = dif.ravel()
    mse = mean((dif)**2)
    return round(mse, 2)

def rmseFunction(test_data, pred_data):
    dif = test_data - pred_data
    dif = dif.ravel()
    mse = mean((dif)**2)
    rmse = sqrt(mse)
    return round(rmse, 2)

def maeFunction(test_data, pred_data):
    dif = test_data - pred_data
    dif = dif.ravel()
    mae = mean(abs(dif))
    return round(mae, 2)

def rsquaredFunction(test_data, pred_data):
    dif = test_data - pred_data
    dif = dif.ravel()
    rsquared = 1 - ( sum((dif) ** 2) / sum((test_data.ravel() - mean(test_data.ravel()))**2))
    return round(rsquared, 5)

def rmsleFunction(test_data, pred_data):
    dif = test_data - pred_data
    dif = dif.ravel()
    mse = mean((dif)**2)
    rmsle = np.log(mse)
    return round(rmsle, 2)
    
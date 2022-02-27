from distutils.log import debug
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import pickle
import numpy as np
import pandas as pd
import numpy as np
from sklearn import linear_model

###################################################
def add_KmeansFeat(record):
    # Kmean features
    Kmean_feats = []
    for i in range(10):
        model_name = 'Kmeans_Features/K-mean-cluster-' + str(i+2) + '.pkl'
        Kmean_feats.append(model_name)

    # get all columns
    cols=record.columns
    
    # create a copy of the record
    copyR = record.copy()
    
    # load the standard scaler pickle
    with open('Kmeans_Features/scalerK.pkl', 'rb') as f:
         scK = pickle.load(f)
    
    # transform data using standard scaler
    copyR.loc[:,cols] = scK.transform(copyR.loc[:,cols])
    
    for i in Kmean_feats:
        # get Kmean feature name
        featName = i.split('/')[1].split('.')[0]
        
        # load the different kmeans pickle
        with open(i, 'rb') as f:
             kmeans = pickle.load(f)
                
        # predict the Kmeans value for that feature
        record[featName] = kmeans.predict(copyR)

def feature_engineering(data):
    # add a new feature to store count number of zeroes across a row
    data['count_zeroes'] = (data == 0).astype(int).sum(axis=1)
    
    # add a new feature to store count number of non zeroes across a rows
    data['count_non_zeroes'] = (data != 0).astype(int).sum(axis=1)
    
    # add a new feature to store if var3 value is mode (most common nationality)
    data['var3_mode'] = [1 if i==2 else 0 for i in data.var3]
    
    # replace -999999 outlier in var3 by mode value
    data.var3 = data.var3.replace(to_replace=-999999, value=2)
    
    # add a new feature to capture customers with var15 below 23
    data['var15_below_23'] = [1 if i<23 else 0 for i in data.var15]
    
    # add a new feature to store the insights for var36 that the value was 99 (outlier)
    data['var36_is_99'] = [1 if i==99 else 0 for i in data.var36]
    
    # replace this 99 in var36 with the value 2 as calculated through KNNImputer
    data.var36 = data.var36.replace(to_replace=99, value=2)
    
    # add a new feature to capture the mode value for var38
    data['var38_mode'] = [1 if i==117310.979016494 else 0 for i in data.var38]
    
    # add a new feature to capture is saldo_medio_var5_ult3 is 0 or not
    data['sal_medio_var5_ult3_is_0'] = [1 if i==0.0 else 0 for i in data.saldo_medio_var5_ult3]
    
    # add a new feature to capture if saldo_var30 is 0 or 3 or something else
    data['sal_var30_is_0_3'] = [1 if i in [0,3] else 0 for i in data.saldo_var30]
    
    # for 'num' keyword features we saw that there was pattern of divisible by 3
    # hence we take sum of those features which are non-zero and divisble by 3 across each row
    numKeywordFeatures = [column for column in data.columns if 'num' in column]
    data['Feat_divisible_by_3']=((data[numKeywordFeatures]%3==0) & (data[numKeywordFeatures]!=0)).astype(int).sum(axis=1)
    
    return data

def get_log_on_data(data):
    # while EDA it was evident to take log on var38 to better Gaussian like distribution
    data['var38_log'] = np.log(data.var38)
    data = data.drop(['var38'], axis=1)
    
    # for features with keyword 'imp' we decided to use the log of the values
    impKeywordFeatures = [column for column in data.columns if 'imp' in column]
    for feat in impKeywordFeatures:
        new_feat = feat + '_log'
        data[new_feat] = [val if val <= 0 else np.log(val) for val in data[feat]]
    data = data.drop(impKeywordFeatures, axis=1)
    
    # for features with keyword 'saldo' we decided to use the log of the values
    saldoKeywordFeatures = [column for column in data.columns if 'saldo' in column]
    for feat in saldoKeywordFeatures:
        new_feat = feat+'_log'
        data[new_feat] = [val if val <= 0 else np.log(val) for val in data[feat]]
    data = data.drop(saldoKeywordFeatures, axis=1)
    
    return data

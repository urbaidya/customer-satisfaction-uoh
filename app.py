from distutils.log import debug
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import pickle
import numpy as np
import joblib
import pandas as pd
import numpy as np
from sklearn import linear_model
from my_function import add_KmeansFeat, feature_engineering, get_log_on_data

# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ## First list of features to remove
    ## (constantFeatures, quasiConstantFeatures and duplicateFeatures)

    with open('all_pickle_files/removeColumns.pkl', 'rb') as f:
        finalFilteredColumns = pickle.load(f)

    ## Second list of features to remove
    ## (highcorrelatedfeatures)
    with open('all_pickle_files/removeCorrColumns.pkl', 'rb') as f:
        featuresWithHighCorr = pickle.load(f)

    to_predict=request.form.to_dict()
    try:
        emp_id=int(to_predict['emp_id'])
    except:
        return flask.render_template('Wrong_entry.html')
    #final_id=emp_id+'.csv'

    try:
        #test_record = pd.read_csv(final_id)
        file = pd.read_csv('database.csv')
        test_record = file[file['ID']==emp_id]
        if test_record.shape[0]!=1:
            return flask.render_template('Wrong_entry.html')
    except:
        return flask.render_template('Wrong_entry.html')
    
    test_record = test_record.drop(['ID'],axis=1)

    # Drop constantFeatures, quasiConstantFeatures and duplicateFeatures
    test_record.drop(finalFilteredColumns,axis=1,inplace=True)

    # Drop High Correlated Features
    test_record.drop(featuresWithHighCorr,axis=1,inplace=True)

    # the Kmeans feature
    add_KmeansFeat(test_record)

    ## do feature engineering 
    test_record1 = feature_engineering(test_record.copy())
    test_record1 = get_log_on_data(test_record1)

    ## get features which needs to be response encoded
    with open('all_pickle_files/features_to_encode_RE.pkl', 'rb') as f:
        features_to_encode_RE = pickle.load(f)

    ## get response encoded dictionary
    with open('all_pickle_files/response_encoded_dict.pkl', 'rb') as f:
        dict_values = pickle.load(f)

    for feat in features_to_encode_RE:
        dict_values_1 = dict_values[feat]['val_1']
        dict_values_0 = dict_values[feat]['val_0']
        uniq_1 = set(dict_values_1.keys())
        uniq_2 = set(dict_values_0.keys())
        feature_1 = feat + '_1'
        feature_0 = feat + '_0'
        
        unique_values = uniq_1.union(uniq_2)
        unique_values.remove('missing')
        
        # get unique values of the column from test  final data
        unique_values_test = set(test_record1[feat].values)
        
        # replace all values which are present in missing from train
        test_record1[feat] = test_record1[feat].apply(lambda x: 'missing' if x in (unique_values_test-unique_values) else x )
        
        test_record1[feature_1] = (test_record1[feat].map(dict_values_1)).values
        test_record1[feature_0] = (test_record1[feat].map(dict_values_0)).values

        test_record1.drop(feat,axis=1,inplace=True)

    ## get response encoded standard scaler
    with open('all_pickle_files/re_sc.pkl', 'rb') as f:
        re_sc = pickle.load(f)

    test_record1_re = re_sc.transform(test_record1)

    ## get response encoded standard scaler
    with open('all_pickle_files/model_xgb4.pkl', 'rb') as f:
        model = pickle.load(f)

    y_pred_val = model.predict_proba(test_record1_re)[:,1]
    if y_pred_val >= 0.038:
        return flask.render_template('Sad.html')
    else:
        return flask.render_template('Happy.html')

    #return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
